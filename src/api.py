"""
Flask backend for the Urban Mobility Lab GUI.
Run with:  python api.py
Then open: http://localhost:5000
"""
import sys, os, threading, uuid, time as _time, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

GUI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'GUI')

import workspace  # sets CWD to project root on import
import nsga2 as _nsga2
import nsga3 as _nsga3
import insecurity
from moad import weighted_astar
from urban_mobility import (
    reconstruct_graph_from_graphml,
    set_elevation_weight,
    dijkstra,
    a_star,
    distance_manhattan,
    distance_euclidean,
    reconstruct_route,
)
import osmnx as ox

app = Flask(__name__)
CORS(app)

# ── Shared state ──────────────────────────────────────────────────────────────
_G = None
_G_lock = threading.Lock()
_jobs: dict = {}
_jobs_lock = threading.Lock()

# ── Graph loading ─────────────────────────────────────────────────────────────

def _load_graph():
    global _G
    with _G_lock:
        if _G is not None:
            return _G, None
        path = workspace.get_graphml_gdl_path()
        if not os.path.exists(path):
            return None, "Graph file not found. Run main.py first."
        _G = reconstruct_graph_from_graphml(path)
        return _G, None

def _ensure_insecurity(G, hour):
    """(Re)computes 'indice_inseg' on the edges when the requested hour changes."""
    with _G_lock:
        if G.graph.get('hora_inseg') != hour:
            insecurity.run(G, None, hora=hour)

def _path_coords(G, path):
    """Convert list of node IDs to [[lat, lon], ...] for Leaflet."""
    return [[float(G.nodes[n]['y']), float(G.nodes[n]['x'])] for n in path]

def _route(G, path):
    """Metrics + coordinates of a path, the same for every algorithm."""
    t, gain, veg = _nsga2.evaluate(G, path)
    length = 0.0
    inseg = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v).get(0, {})
        length += float(data.get('length', 0.0))
        inseg  += float(data.get('indice_inseg', 0.0))
    edges = max(len(path) - 1, 1)
    return {'time_s': round(t, 1), 'elev_gain_m': round(gain, 2),
            'veg_cost': round(veg, 3), 'insecurity': round(inseg / edges, 3),
            'length_m': round(length, 1), 'nodes': len(path),
            'coords': _path_coords(G, path)}

# ── Job management ─────────────────────────────────────────────────────────────

def _new_job():
    jid = str(uuid.uuid4())[:8]
    with _jobs_lock:
        _jobs[jid] = {'status': 'running', 'log': [], 'result': None, 'error': None}
    return jid

def _log(jid, msg):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid]['log'].append(msg)

def _done(jid, result):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid]['status'] = 'done'
            _jobs[jid]['result'] = result

def _fail(jid, msg):
    with _jobs_lock:
        if jid in _jobs:
            _jobs[jid]['status'] = 'error'
            _jobs[jid]['error'] = msg

# ── Stdout capture ────────────────────────────────────────────────────────────

class _JobStream:
    """Captures print() output and forwards each line to the job log."""
    def __init__(self, jid, original):
        self.jid = jid
        self.original = original
        self._buf = ''

    def write(self, text):
        self.original.write(text)           # still print to terminal
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if line.strip():
                _log(self.jid, line)

    def flush(self):
        self.original.flush()
        if self._buf.strip():
            _log(self.jid, self._buf.strip())
            self._buf = ''

    def isatty(self):
        return False


# ── Algorithm runners (run in background threads) ─────────────────────────────

def _capture(jid):
    """Context manager: redirects print() to the job log."""
    return contextlib.redirect_stdout(_JobStream(jid, sys.stdout))


def _run_dijkstra(jid, elevation, start_node, end_node):
    try:
        G, err = _load_graph()
        if err:
            return _fail(jid, err)
        weight = "ele_diff" if elevation else "length"
        if elevation:
            set_elevation_weight(G)
        with _capture(jid):
            dist, prev = dijkstra(G, start_node, weight=weight)
        path = reconstruct_route(prev, start_node, end_node)
        if not path:
            return _fail(jid, "No path found between the selected nodes.")
        _done(jid, {'type': 'single', 'algorithm': 'Dijkstra', 'route': _route(G, path)})
    except Exception as e:
        _fail(jid, str(e))


def _run_astar(jid, heuristic_name, elevation, start_node, end_node):
    try:
        G, err = _load_graph()
        if err:
            return _fail(jid, err)
        if elevation:
            set_elevation_weight(G)
        workspace.set_elevation_flag(elevation)
        heuristic = distance_manhattan if heuristic_name == 'manhattan' else distance_euclidean
        label = f'A* {heuristic_name.capitalize()}'
        with _capture(jid):
            dist, prev = a_star(G, start_node, end_node, heuristic, weight_d="elevation")
        path = reconstruct_route(prev, start_node, end_node)
        if not path:
            return _fail(jid, "No path found between the selected nodes.")
        _done(jid, {'type': 'single', 'algorithm': label, 'route': _route(G, path)})
    except Exception as e:
        _fail(jid, str(e))


def _run_nsga(jid, version, pop_size, generations, mutation_rate, divisions, start_node, end_node):
    try:
        G, err = _load_graph()
        if err:
            return _fail(jid, err)
        label = 'NSGA-II' if version == 2 else 'NSGA-III'
        t0 = _time.perf_counter()
        with _capture(jid):
            if version == 2:
                pareto = _nsga2.nsga2(G, start_node, end_node,
                                      pop_size=pop_size, generations=generations,
                                      mutation_rate=mutation_rate)
            else:
                pareto = _nsga3.nsga3(G, start_node, end_node,
                                      pop_size=pop_size, generations=generations,
                                      mutation_rate=mutation_rate, divisions=divisions)
        elapsed = round(_time.perf_counter() - t0, 1)
        solutions = [_route(G, p) for p, _ in pareto]
        _done(jid, {'type': 'pareto', 'algorithm': label,
                    'solutions': solutions, 'compute_time_s': elapsed})
    except Exception as e:
        _fail(jid, str(e))


def _run_compare(jid, pop_size, generations, mutation_rate, divisions, start_node, end_node):
    try:
        G, err = _load_graph()
        if err:
            return _fail(jid, err)

        t0 = _time.perf_counter()
        with _capture(jid):
            pareto2 = _nsga2.nsga2(G, start_node, end_node,
                                   pop_size=pop_size, generations=generations,
                                   mutation_rate=mutation_rate)
        time2 = round(_time.perf_counter() - t0, 1)

        t0 = _time.perf_counter()
        with _capture(jid):
            pareto3 = _nsga3.nsga3(G, start_node, end_node,
                                   pop_size=pop_size, generations=generations,
                                   mutation_rate=mutation_rate, divisions=divisions)
        time3 = round(_time.perf_counter() - t0, 1)

        _done(jid, {
            'type': 'compare',
            'nsga2': {'solutions': [_route(G, p) for p, _ in pareto2], 'compute_time_s': time2},
            'nsga3': {'solutions': [_route(G, p) for p, _ in pareto3], 'compute_time_s': time3},
        })
    except Exception as e:
        _fail(jid, str(e))


def _run_weighted_astar(jid, weights, start_node, end_node):
    try:
        G, err = _load_graph()
        if err:
            return _fail(jid, err)
        t0 = _time.perf_counter()
        with _capture(jid):
            print("Weighted A*  w_time=%.2f  w_elev=%.2f  w_veg=%.2f  w_sec=%.2f" % weights)
            path = weighted_astar(G, start_node, end_node, *weights)
        if not path:
            return _fail(jid, "No path found between the selected nodes.")
        _done(jid, {'type': 'single', 'algorithm': 'Weighted A*', 'route': _route(G, path),
                    'compute_time_s': round(_time.perf_counter() - t0, 1)})
    except Exception as e:
        _fail(jid, str(e))


# ── HTTP endpoints ────────────────────────────────────────────────────────────

@app.route('/')
@app.route('/lab')
def serve_lab():
    return send_from_directory(GUI_DIR, 'lab.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(GUI_DIR, filename)

@app.route('/api/graph/status')
def graph_status():
    exists = os.path.exists(workspace.get_graphml_gdl_path())
    return jsonify({'exists': exists})


_grid_cache: dict = {}

@app.route('/api/insecurity')
def insecurity_grid():
    """500 m cells with their final SI (0-100) at the requested hour, for the map layer."""
    hour = int(request.args.get('hour', 12)) % 24
    if hour not in _grid_cache:
        cells = insecurity.cargar_modelo()['si_calculado'].index.droplevel('hora').unique()
        ix = cells.get_level_values(0).to_numpy()
        iy = cells.get_level_values(1).to_numpy()
        dlat = insecurity.TAM_CELDA / insecurity.KM_POR_GRADO_LAT
        dlon = insecurity.TAM_CELDA / insecurity.KM_POR_GRADO_LON
        lat0 = insecurity.LAT0 + iy * dlat   # south-west corner
        lon0 = insecurity.LON0 + ix * dlon
        si = insecurity.si_en(lat0 + dlat / 2, lon0 + dlon / 2, hour)
        _grid_cache[hour] = {
            'hour': hour, 'dlat': dlat, 'dlon': dlon,
            'cells': [[round(float(a), 6), round(float(o), 6), round(float(v), 1)]
                      for a, o, v in zip(lat0, lon0, si)],
        }
    return jsonify(_grid_cache[hour])


@app.route('/api/run', methods=['POST'])
def run():
    data = request.json or {}
    algo      = data.get('algorithm', 'nsga2')
    elevation = bool(data.get('elevation', True))
    params    = data.get('params', {})
    start_lat = float(data.get('start_lat', 20.630163))
    start_lon = float(data.get('start_lon', -103.376624))
    end_lat   = float(data.get('end_lat',   20.697814))
    end_lon   = float(data.get('end_lon',  -103.384384))
    hour      = int(data.get('hour', 12)) % 24
    weights   = tuple(max(0.0, min(1.0, float(params.get(k, d))))
                      for k, d in (('w_time', 0.5), ('w_elev', 0.0), ('w_veg', 0.0), ('w_sec', 0.5)))

    G, err = _load_graph()
    if err:
        return jsonify({'error': err}), 400
    _ensure_insecurity(G, hour)

    start_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
    end_node   = ox.distance.nearest_nodes(G, end_lon,   end_lat)

    pop_size      = max(10, int(params.get('pop_size', 60)))
    generations   = max(5,  int(params.get('generations', 20)))
    mutation_rate = max(0.0, min(1.0, float(params.get('mutation_rate', 0.15))))
    divisions     = max(2,  int(params.get('divisions', 12)))

    jid = _new_job()

    if algo == 'weighted_astar':
        target = (_run_weighted_astar, (jid, weights, start_node, end_node))
    elif algo == 'dijkstra':
        target = (_run_dijkstra, (jid, elevation, start_node, end_node))
    elif algo == 'astar_manhattan':
        target = (_run_astar, (jid, 'manhattan', elevation, start_node, end_node))
    elif algo == 'astar_euclidean':
        target = (_run_astar, (jid, 'euclidean', elevation, start_node, end_node))
    elif algo == 'nsga2':
        target = (_run_nsga, (jid, 2, pop_size, generations, mutation_rate, divisions, start_node, end_node))
    elif algo == 'nsga3':
        target = (_run_nsga, (jid, 3, pop_size, generations, mutation_rate, divisions, start_node, end_node))
    elif algo == 'compare':
        target = (_run_compare, (jid, pop_size, generations, mutation_rate, divisions, start_node, end_node))
    else:
        return jsonify({'error': f'Unknown algorithm: {algo}'}), 400

    threading.Thread(target=target[0], args=target[1], daemon=True).start()
    return jsonify({'job_id': jid})


@app.route('/api/status/<jid>')
def status(jid):
    with _jobs_lock:
        job = dict(_jobs.get(jid, {}))
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({'status': job['status'], 'log': list(job['log'])})


@app.route('/api/result/<jid>')
def result(jid):
    with _jobs_lock:
        job = dict(_jobs.get(jid, {}))
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job['status'] == 'error':
        return jsonify({'error': job['error']}), 500
    if job['status'] != 'done':
        return jsonify({'error': 'Not ready yet'}), 202
    return jsonify(job['result'])


if __name__ == '__main__':
    print("Urban Mobility API  →  http://localhost:5000")
    print("Open http://localhost:5000 in your browser to start.")
    app.run(host='localhost', port=5000, debug=False, threaded=True)
