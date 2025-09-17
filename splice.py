#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Splice (pygame edition) — a 2-player smoothing game on a random immersed closed curve.

How to run
----------
$ python splice.py

Requires: Python 3.9+, numpy, pygame.

Controls
--------
- Left-click near a crossing dot to cycle X → A → B → X (pending only).
- Confirm move: click the "Confirm" button.
- New random curve: click "New".
- Undo last move: click "Undo".
- Quit: press ESC or close the window.

Notes
-----
- This version ports the matplotlib rendering to pygame. It preserves the improved scoring:
  a loop scores only if it is a simple cycle that does not traverse any 'X' crossing (a Jordan loop).
  The "outside disk" is awarded only when all crossings are smoothed and there is exactly one Jordan loop.
- The generator supports constraints:
  - min_cross_sep: minimum Euclidean distance between distinct crossings (world units).
  - min_cross_angle_deg: minimum acute angle (in degrees) between the two strands at every crossing.

Enjoy!
"""
import math
import random
import sys
from collections import defaultdict, namedtuple

import asyncio

import numpy as np
import pygame

# --------------------------- Configuration ------------------------------------

WINDOW_W, WINDOW_H = 900, 900
UI_H = 80  # bottom UI bar height
BG_COLOR = (250, 250, 252)
EDGE_COLOR = (30, 30, 30)
SMOOTH_COLOR = (220, 30, 30)   # red for crossing neighborhoods (X/A/B)
CROSS_DOT_COLOR = (20, 20, 20)
CLAIM_FILL_COLOR_P1 = (80, 140, 255, 60)   # RGBA with alpha
CLAIM_FILL_COLOR_P2 = (255, 140, 80, 60)
TEXT_COLOR = (15, 15, 18)
BTN_TEXT = (20, 20, 22)
BTN_BG = (235, 235, 240)
BTN_BG_HOVER = (215, 215, 225)
BTN_BORDER = (180, 180, 190)
PENDING_HALO = (20, 120, 255)

EDGE_W = 2           # stroke width for straight segments
SMOOTH_W = EDGE_W              # same thickness as the base curve
FILL_SAFETY_PX = 3.0 # extra px to stop straight segments short of the join

NEIGH_W = max(EDGE_W, SMOOTH_W) + 2  # thickness of the neighborhood “tube”

GLOBAL_PORT_RADIUS_PX = 24

FPS = 60

# --------------------------- Utilities ----------------------------------------

def set_seed(seed=None):
    if seed is None:
        seed = random.randrange(1 << 30)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    return seed

def angle_of(vec):
    return float(np.arctan2(vec[1], vec[0]))

def shoelace_area(poly):
    if len(poly) < 3:
        return 0.0
    x = np.array([p[0] for p in poly])
    y = np.array([p[1] for p in poly])
    x2 = np.append(x, x[0])
    y2 = np.append(y, y[0])
    return 0.5 * float(np.dot(x2[:-1], y2[1:]) - np.dot(y2[:-1], x2[1:]))

def point_in_poly(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-16) + x1
            if xinters > x:
                inside = not inside
    return inside

def seg_intersection(p, p2, q, q2, tol=1e-9):
    r = p2 - p
    s = q2 - q
    rxs = r[0] * s[1] - r[1] * s[0]
    qmp = q - p
    if abs(rxs) < tol:
        return None
    t = (qmp[0] * s[1] - qmp[1] * s[0]) / rxs
    u = (qmp[0] * r[1] - qmp[1] * r[0]) / rxs
    if tol < t < 1 - tol and tol < u < 1 - tol:
        pt = p + t * r
        return t, u, pt
    return None

def polygon_centroid(poly):
    A = shoelace_area(poly)
    if abs(A) < 1e-12:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx = sum((poly[i][0] + poly[(i+1)%len(poly)][0]) *
             (poly[i][0]*poly[(i+1)%len(poly)][1] - poly[(i+1)%len(poly)][0]*poly[i][1])
             for i in range(len(poly))) / (6*A)
    cy = sum((poly[i][1] + poly[(i+1)%len(poly)][1]) *
             (poly[i][0]*poly[(i+1)%len(poly)][1] - poly[(i+1)%len(poly)][0]*poly[i][1])
             for i in range(len(poly))) / (6*A)
    return (cx, cy)

def interior_point(poly, tries=200):
    """Return a point guaranteed (with high probability) to lie inside `poly` by random sampling its bbox."""
    if not poly:
        return (0.0, 0.0)
    minx = min(p[0] for p in poly); maxx = max(p[0] for p in poly)
    miny = min(p[1] for p in poly); maxy = max(p[1] for p in poly)
    for _ in range(tries):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        if point_in_poly((x, y), poly):
            return (x, y)
    # Fallback (very rare): use area centroid
    return polygon_centroid(poly)

# --------------------- Random curve & immersed graph ---------------------------

def random_fourier_closed_polyline(M=140, modes=4, radius=5.0):
    t = np.linspace(0, 2*np.pi, M, endpoint=False)
    def coeffs():
        a = np.random.normal(0, 0.8, size=modes+1)
        b = np.random.normal(0, 0.8, size=modes+1)
        a[0] = 0.0
        b[0] = 0.0
        scale = 1/np.arange(1, modes+2)**1.3
        return a*scale, b*scale
    ax, bx = coeffs()
    ay, by = coeffs()
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    for k in range(1, modes+1):
        x += ax[k]*np.cos(k*t) + bx[k]*np.sin(k*t)
        y += ay[k]*np.cos(k*t) + by[k]*np.sin(k*t)
    x = radius * (x - np.mean(x)) / (np.std(x) + 1e-9)
    y = radius * (y - np.mean(y)) / (np.std(y) + 1e-9)
    pts = np.stack([x, y], axis=1)
    return pts

def build_immersed_graph(poly, sep_min=None, angle_min_rad=None, max_attempts=6):
    N = len(poly)
    segs = []
    for i in range(N):
        a = poly[i]
        b = poly[(i+1) % N]
        segs.append((i, (a, b)))

    seg_to_splits = defaultdict(list)
    nodes = []
    node_id_of_point = {}
    tol = 1e-7

    def key_of_pt(pt):
        return (round(float(pt[0]) / tol) * tol, round(float(pt[1]) / tol) * tol)

    for i in range(N):
        p = tuple(map(float, poly[i]))
        nid = len(nodes)
        nodes.append({'pos': p, 'type': 'vertex', 'inc': [], 'angles': []})
        node_id_of_point[('v', i)] = nid

    for i in range(N):
        a, b = segs[i][1]
        for j in range(i+1, N):
            if j == i or j == (i+1) % N or i == (j+1) % N:
                continue
            c, d = segs[j][1]
            res = seg_intersection(np.array(a), np.array(b), np.array(c), np.array(d))
            if res is None:
                continue
            t, u, pt = res
            seg_to_splits[i].append((t, pt))
            seg_to_splits[j].append((u, pt))
            k = key_of_pt(pt)
            if ('x', k) not in node_id_of_point:
                nid = len(nodes)
                nodes.append({'pos': (float(pt[0]), float(pt[1])), 'type': 'cross', 'inc': [], 'angles': []})
                node_id_of_point[('x', k)] = nid

    Edge = namedtuple('Edge', 'id u v p_u p_v')
    edges = []

    def node_id_for_segment_endpoint(seg_idx, is_end):
        vid = seg_idx if not is_end else (seg_idx + 1) % N
        return node_id_of_point[('v', vid)]

    for i in range(N):
        a = segs[i][1][0]
        b = segs[i][1][1]
        splits = seg_to_splits.get(i, [])
        enriched = [(0.0, np.array(a), node_id_for_segment_endpoint(i, False))] + \
                   [(t, np.array(pt), None) for (t, pt) in splits] + \
                   [(1.0, np.array(b), node_id_for_segment_endpoint(i, True))]
        enriched.sort(key=lambda x: x[0])
        for k in range(len(enriched) - 1):
            t1, p1, nid1 = enriched[k]
            t2, p2, nid2 = enriched[k+1]
            if nid1 is None:
                nid1 = node_id_of_point[('x', key_of_pt(p1))]
            if nid2 is None:
                nid2 = node_id_of_point[('x', key_of_pt(p2))]
            eid = len(edges)
            edges.append(Edge(eid, nid1, nid2, tuple(map(float, p1)), tuple(map(float, p2))))
            nodes[nid1]['inc'].append(eid)
            nodes[nid2]['inc'].append(eid)

    bad = [i for i, n in enumerate(nodes) if n['type'] == 'cross' and len(n['inc']) != 4]
    if bad:
        if max_attempts > 0:
            return None
        else:
            raise RuntimeError("Degenerate crossing with valence !=4: " + str(bad))

    for nid, n in enumerate(nodes):
        n['angles'] = []
        for eid in n['inc']:
            e = edges[eid]
            if e.u == nid:
                other_pt = np.array(e.p_v)
                here_pt = np.array(e.p_u)
            else:
                other_pt = np.array(e.p_u)
                here_pt = np.array(e.p_v)
            vec = other_pt - here_pt
            ang = angle_of(vec)
            n['angles'].append((eid, ang))
        n['angles'].sort(key=lambda t: t[1])

    # Constraints
    if sep_min is not None or angle_min_rad is not None:
        cross_ids = [i for i, n in enumerate(nodes) if n['type'] == 'cross']
        if cross_ids:
            if sep_min is not None:
                pts = np.array([nodes[i]['pos'] for i in cross_ids], dtype=float)
                dmin = float('inf')
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        d = math.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
                        if d < dmin:
                            dmin = d
                if dmin < float(sep_min):
                    return None
            if angle_min_rad is not None:
                for nid in cross_ids:
                    angs = [a for (_, a) in nodes[nid]['angles']]
                    angs = [((a + math.pi) % (2*math.pi)) - math.pi for a in angs]
                    dirs = [(a + math.pi) % math.pi for a in angs]
                    dirs.sort()
                    tol = 1e-4
                    uniq = []
                    for d in dirs:
                        if not any(abs(d - u) < tol for u in uniq):
                            uniq.append(d)
                    if len(uniq) != 2:
                        return None
                    delta = abs(uniq[0] - uniq[1])
                    delta = min(delta, math.pi - delta)
                    if delta < float(angle_min_rad):
                        return None

    return nodes, edges

# -------------------------- Core game model ------------------------------------

class SpliceGame:
    def __init__(self, seed=None, min_cross_sep=0.6, min_cross_angle_deg=25.0):
        self.min_cross_sep = min_cross_sep
        self.min_cross_angle_rad = math.radians(min_cross_angle_deg)
        self.seed = set_seed(seed)
        self.nodes, self.edges = self._make_good_random_graph()
        self._build_halfedges()
        self.cross_state = {nid: 'X' for nid, n in enumerate(self.nodes) if n['type'] == 'cross'}
        self.pending_cross = None
        self.player = 0
        self.scores = [0, 0]
        self.claimed = []  # {'nodes', 'poly', 'centroid', 'owner', 'outside'}
        self.seen_simple_cycles = set()
        self._update_seen_cycles()
        self.port_radius_world = None  # set by the renderer once it knows the scale


    def _make_good_random_graph(self):
        for _ in range(256):
            # Scale up if min_cross_sep is large:
            r = max(5.0, 2.0 * self.min_cross_sep)
            poly = random_fourier_closed_polyline(M=140, modes=4, radius=r)
            res = build_immersed_graph(poly, sep_min=self.min_cross_sep,
                                       angle_min_rad=self.min_cross_angle_rad)
            if res is not None:
                nodes, edges = res
                n_cross = sum(1 for n in nodes if n['type'] == 'cross')
                if n_cross >= 6:
                    return nodes, edges
        raise RuntimeError("Could not generate a curve meeting spacing/angle constraints; loosen parameters or retry.")

    def _build_halfedges(self):
        self.stub_to_he = {nid: {} for nid in range(len(self.nodes))}
        self.halfedges = []
        self.edge_to_he = {}
        for e in self.edges:
            h1 = {'eid': e.id, 'tail': e.u, 'head': e.v}
            h2 = {'eid': e.id, 'tail': e.v, 'head': e.u}
            h1_id = len(self.halfedges); self.halfedges.append(h1)
            h2_id = len(self.halfedges); self.halfedges.append(h2)
            self.edge_to_he[(e.id, 0)] = h1_id
            self.edge_to_he[(e.id, 1)] = h2_id
        self.edge_endpoint_stub = {}
        for nid, n in enumerate(self.nodes):
            inc_sorted = n['angles']
            for idx, (eid, ang) in enumerate(inc_sorted):
                e = self.edges[eid]
                if e.u == nid:
                    he_id = self.edge_to_he[(eid, 0)]
                else:
                    he_id = self.edge_to_he[(eid, 1)]
                self.stub_to_he[nid][idx] = he_id
        for nid, n in enumerate(self.nodes):
            for idx, (eid, ang) in enumerate(n['angles']):
                self.edge_endpoint_stub[(eid, nid)] = idx

    def _next_edge_along_strand(self, edge, at_node):
        """Continue along the same strand through a vertex/crossing (world space)."""
        idx_in = self.edge_endpoint_stub[(edge.id, at_node)]
        node = self.nodes[at_node]
        if node['type'] == 'vertex':     # degree 2
            out_idx = 1 - idx_in
        else:                             # crossing: go straight across
            out_idx = (idx_in + 2) % 4
        eid2 = node['angles'][out_idx][0]
        e2 = self.edges[eid2]
        if e2.u == at_node:
            p0 = np.array(self.nodes[at_node]['pos'], float)
            p1 = np.array(e2.p_v, float)
            nxt = e2.v
        else:
            p0 = np.array(self.nodes[at_node]['pos'], float)
            p1 = np.array(e2.p_u, float)
            nxt = e2.u
        return e2, p0, p1, nxt
    
    def _point_seg_dist(self, p, a, b):
        """Euclidean distance from point p to segment ab (all world coords)."""
        px, py = p
        ax, ay = a
        bx, by = b
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        vv = vx*vx + vy*vy
        if vv <= 1e-20:
            dx, dy = px - ax, py - ay
            return math.hypot(dx, dy)
        t = max(0.0, min(1.0, (wx*vx + wy*vy) / vv))
        cx = ax + t * vx
        cy = ay + t * vy
        return math.hypot(px - cx, py - cy)

    def _distance_to_edges(self, p, poly):
        """Minimum distance (world units) from point p to any polygon edge."""
        n = len(poly)
        mind = float('inf')
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            d = self._point_seg_dist(p, a, b)
            if d < mind:
                mind = d
        return mind

    def _best_label_point(self, poly, margin_world=None, grid=12, refine_steps=(0.5, 0.25, 0.12)):
        """
        Approximate pole of inaccessibility: pick the point inside `poly` with
        maximum distance to its edges. `margin_world` asks for at least that clearance;
        if not achievable, we return the deepest point anyway.
        """
        if not poly:
            return (0.0, 0.0)

        # bbox
        minx = min(p[0] for p in poly); maxx = max(p[0] for p in poly)
        miny = min(p[1] for p in poly); maxy = max(p[1] for p in poly)
        dx = maxx - minx or 1.0
        dy = maxy - miny or 1.0

        best_pt = None
        best_d  = -1.0

        # include centroid if inside
        c0 = polygon_centroid(poly)
        if point_in_poly(c0, poly):
            d0 = self._distance_to_edges(c0, poly)
            best_pt, best_d = c0, d0

        # coarse grid search
        for i in range(grid):
            for j in range(grid):
                x = minx + (i + 0.5) * dx / grid
                y = miny + (j + 0.5) * dy / grid
                pt = (x, y)
                if point_in_poly(pt, poly):
                    d = self._distance_to_edges(pt, poly)
                    if d > best_d:
                        best_pt, best_d = pt, d

        # local refinement around the current best
        if best_pt is not None and best_d > 0.0:
            for frac in refine_steps:
                radius = best_d * frac
                # try a ring of candidates
                for k in range(20):
                    ang = 2.0 * math.pi * (k / 20.0 + random.random() * 0.02)
                    pt = (best_pt[0] + radius * math.cos(ang),
                        best_pt[1] + radius * math.sin(ang))
                    if point_in_poly(pt, poly):
                        d = self._distance_to_edges(pt, poly)
                        if d > best_d:
                            best_pt, best_d = pt, d

        # if nothing worked (degenerate), fall back to random interior sample
        if best_pt is None:
            best_pt = interior_point(poly)
            best_d = self._distance_to_edges(best_pt, poly)

        # If we were asked for a margin and can't meet it, we'll still return the deepest point.
        return (float(best_pt[0]), float(best_pt[1]))


    def _port_on_edge_world(self, cross_id, neighbor_id, Rw, max_steps=400):
        """
        Return (point, unit_dir) for the first intersection of the strand that leaves
        crossing `cross_id` toward `neighbor_id` with the circle of radius Rw centered
        at the crossing. Walks across short edges/vertices exactly like the renderer.
        """
        Cw = np.array(self.nodes[cross_id]['pos'], float)

        # find the edge that connects cross_id <-> neighbor_id
        eid0 = None
        for eid in self.nodes[cross_id]['inc']:
            e = self.edges[eid]
            if (e.u == neighbor_id) or (e.v == neighbor_id):
                eid0 = eid
                break
        if eid0 is None:
            # extremely defensive; should not happen
            return (float(Cw[0]) + Rw, float(Cw[1])), (1.0, 0.0)

        e = self.edges[eid0]
        if e.u == cross_id:
            p0 = np.array(e.p_u, float)   # == Cw
            p1 = np.array(e.p_v, float)
            cur_node = e.v
        else:
            p0 = np.array(e.p_v, float)   # == Cw
            p1 = np.array(e.p_u, float)
            cur_node = e.u

        steps = 0
        while steps < max_steps:
            a = p1 - p0
            A = float(np.dot(a, a))
            b = p0 - Cw
            B = 2.0 * float(np.dot(a, b))
            C = float(np.dot(b, b) - Rw*Rw)
            disc = B*B - 4*A*C

            if disc >= 0.0 and A > 1e-20:
                sq = math.sqrt(disc)
                t1 = (-B - sq) / (2*A)
                t2 = (-B + sq) / (2*A)
                cands = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
                if cands:
                    t = min([t for t in cands if t >= 1e-9] or cands)
                    Pw = p0 + t * a
                    u = a / (np.linalg.norm(a) + 1e-12)
                    return (float(Pw[0]), float(Pw[1])), (float(u[0]), float(u[1]))

            # segment is entirely inside the circle; continue along the strand
            e, p0, p1, cur_node = self._next_edge_along_strand(e, cur_node)
            steps += 1

        # fallback (should never be reached)
        u = p1 - p0
        L = float(np.linalg.norm(u)) or 1.0
        u /= L
        Pw = Cw + Rw * u
        return (float(Pw[0]), float(Pw[1])), (float(u[0]), float(u[1]))


    @staticmethod
    def _pairing_for_cross(deg, state):
        assert deg == 4
        if state == 'X':
            return {0: 2, 2: 0, 1: 3, 3: 1}
        elif state == 'A':
            return {0: 1, 1: 0, 2: 3, 3: 2}
        elif state == 'B':
            return {1: 2, 2: 1, 3: 0, 0: 3}
        else:
            raise ValueError("Unknown state")

    @staticmethod
    def _pairing_for_vertex(deg):
        assert deg == 2
        return {0: 1, 1: 0}

    def _node_pairing(self, nid, use_pending=False):
        n = self.nodes[nid]
        deg = len(n['inc'])
        if n['type'] == 'vertex':
            return self._pairing_for_vertex(deg)
        state = self.cross_state[nid]
        if use_pending and self.pending_cross is not None and self.pending_cross[0] == nid:
            state = self.pending_cross[1]
        return self._pairing_for_cross(deg, state)

    def successor(self, he_id, use_pending=False):
        h = self.halfedges[he_id]
        head = h['head']
        eid = h['eid']
        s_in = self.edge_endpoint_stub[(eid, head)]
        pairing = self._node_pairing(head, use_pending=use_pending)
        s_out = pairing[s_in]
        he_next = self.stub_to_he[head][s_out]
        return he_next

    def enumerate_cycles(self, use_pending=False):
        visited = set()
        cycles = []
        H = len(self.halfedges)
        for h0 in range(H):
            if h0 in visited:
                continue
            seq_h = []
            seq_nodes = []
            h = h0
            while True:
                if h in visited:
                    break
                visited.add(h)
                seq_h.append(h)
                he = self.halfedges[h]
                head = he['head']
                seq_nodes.append(head)
                h = self.successor(h, use_pending=use_pending)
                if h == h0:
                    seq_nodes.append(self.halfedges[h]['tail'])
                    break
            nodes_no_last = seq_nodes[:-1]
            simple = (len(set(nodes_no_last)) == len(nodes_no_last))
            poly = [self.nodes[nid]['pos'] for nid in nodes_no_last]
            area = shoelace_area(poly)
            cycles.append({'halfedges': seq_h, 'nodes': seq_nodes, 'simple': simple, 'poly': poly, 'area': area})
        return cycles

    @staticmethod
    def _canon_cycle_nodes(nodes):
        if not nodes:
            return tuple()
        if nodes[0] == nodes[-1]:
            nodes = nodes[:-1]
        n = len(nodes)
        candidates = []
        for rev in [False, True]:
            seq = nodes[::-1] if rev else nodes[:]
            min_idx = min(range(n), key=lambda i: seq[i])
            rot = seq[min_idx:] + seq[:min_idx]
            candidates.append(tuple(rot))
        return min(candidates)

    def is_jordan_cycle(self, c, use_pending=False):
        """
        A cycle is Jordan iff:
        (i) its boundary does not traverse any 'X' crossing, and
        (ii) the rendered (black+red) boundary encloses nonzero area.

        IMPORTANT: We do NOT require node-simplicity. A valid region can revisit the
        same smoothed crossing twice (two red arcs from one disk) and still be a
        simple closed curve geometrically.
        """
        # (i) Can't traverse any unsmoothed crossings
        for nid in set(c['nodes'][:-1]):
            node = self.nodes[nid]
            if node['type'] == 'cross':
                state = self.cross_state[nid]
                if use_pending and self.pending_cross is not None and self.pending_cross[0] == nid:
                    state = self.pending_cross[1]
                if state == 'X':
                    return False

        # (ii) Build the exact boundary we render and check area
        r_map = self._cross_radii()
        poly_sm = self._smoothed_cycle_poly(c['nodes'][:-1], r_map)

        if len(poly_sm) < 3:
            return False
        if abs(shoelace_area(poly_sm)) <= 1e-9:
            return False

        c['poly_sm'] = poly_sm  # cache for scoring/painting
        return True




    def _update_seen_cycles(self):
        cycles = self.enumerate_cycles(use_pending=False)
        self.seen_simple_cycles = set()
        for c in cycles:
            if self.is_jordan_cycle(c):
                key = self._canon_cycle_nodes(c['nodes'][:-1])
                self.seen_simple_cycles.add(key)

    def commit_and_score(self):
        if self.pending_cross is None:
            return {'moved': False, 'awards': []}

        nid, new_state = self.pending_cross
        self.cross_state[nid] = new_state
        self.pending_cross = None

        # NEW: radii for building smoothed polygons in the model
        r_map = self._cross_radii()

        cycles_after = self.enumerate_cycles(use_pending=False)

        new_jordan = []
        for c in cycles_after:
            if self.is_jordan_cycle(c):
                key = self._canon_cycle_nodes(c['nodes'][:-1])
                if key not in self.seen_simple_cycles:
                    # Attach a smoothed polygon for region tests
                    c['poly_sm'] = self._smoothed_cycle_poly(c['nodes'][:-1], r_map)
                    new_jordan.append(c)

        def contains_claimed(poly_sm):
            for cl in self.claimed:
                if point_in_poly(cl['label'], poly_sm):
                    return True
            return False

        new_awards = []

        for c in new_jordan:
            poly_sm = c.get('poly_sm') or self._smoothed_cycle_poly(c['nodes'][:-1], r_map)
            if len(poly_sm) >= 3 and not contains_claimed(poly_sm):
                # Choose a pixel-based safety margin and convert to world units
                desired_margin_px = 12.0
                if self.port_radius_world is not None:
                    margin_world = (desired_margin_px / float(GLOBAL_PORT_RADIUS_PX)) * float(self.port_radius_world)
                else:
                    margin_world = 0.5  # conservative fallback

                lbl = self._best_label_point(poly_sm, margin_world=margin_world)
                self.claimed.append({'nodes': c['nodes'],
                                    'poly': poly_sm,     # store the smoothed outline we used
                                    'label': lbl,
                                    'owner': self.player,
                                    'outside': False})
                self.scores[self.player] += 1
                new_awards.append({'type': 'disk', 'area': shoelace_area(poly_sm), 'player': self.player})

        cross_ids = [i for i, n in enumerate(self.nodes) if n['type'] == 'cross']
        if all(self.cross_state[cid] != 'X' for cid in cross_ids):
            jordan_after = [c for c in cycles_after if self.is_jordan_cycle(c)]
            if len(jordan_after) == 1:
                onlyc = jordan_after[0]
                already_claimed_exact = any(
                    self._canon_cycle_nodes(cl['nodes'][:-1]) ==
                    self._canon_cycle_nodes(onlyc['nodes'][:-1])
                    for cl in self.claimed
                )
                if not already_claimed_exact:
                    poly_sm = self._smoothed_cycle_poly(onlyc['nodes'][:-1], r_map)
                    if len(poly_sm) >= 3:
                        # Choose a pixel-based safety margin and convert to world units
                        desired_margin_px = 12.0
                        if self.port_radius_world is not None:
                            margin_world = (desired_margin_px / float(GLOBAL_PORT_RADIUS_PX)) * float(self.port_radius_world)
                        else:
                            margin_world = 0.5  # conservative fallback

                        lbl = self._best_label_point(poly_sm, margin_world=margin_world)
                        self.claimed.append({'nodes': onlyc['nodes'],
                                            'poly': poly_sm,
                                            'label': lbl,
                                            'owner': self.player,
                                            'outside': True})
                        self.scores[self.player] += 1
                        new_awards.append({'type': 'outside', 'area': shoelace_area(poly_sm), 'player': self.player})

        self._update_seen_cycles()
        self.player ^= 1
        return {'moved': True, 'awards': new_awards}

    def find_nearest_crossing(self, sx, sy, world_to_screen, max_pix=14):
        best = None
        best_d = float('inf')
        for nid, n in enumerate(self.nodes):
            if n['type'] != 'cross':
                continue
            # Only allow crossings that are still singular (not yet smoothed/committed)
            if self.cross_state[nid] != 'X':
                continue
            x, y = n['pos']
            px, py = world_to_screen((x, y))
            d = math.hypot(px - sx, py - sy)
            if d < best_d and d <= max_pix:
                best_d = d
                best = nid
        return best

    def cycle_crossing_state(self, nid):
        cur = self.cross_state[nid]
        if self.pending_cross is not None and self.pending_cross[0] == nid:
            cur = self.pending_cross[1]
        nxt = {'X': 'A', 'A': 'B', 'B': 'X'}[cur]
        if nxt != self.cross_state[nid]:
            self.pending_cross = (nid, nxt)
        else:
            self.pending_cross = None
    
    def _cross_radii(self):
        """
        World radii per crossing that match the renderer’s red ports.
        If the renderer hasn’t told us the scale yet, fall back to the old heuristic
        so the game can still initialize; the renderer will refresh seen cycles once ready.
        """
        if self.port_radius_world is not None:
            R = float(self.port_radius_world)
            return {nid: R for nid, n in enumerate(self.nodes) if n['type'] == 'cross'}

        # --- Fallback (initial boot before renderer sets port_radius_world) ---
        r_map = {}
        for nid, n in enumerate(self.nodes):
            if n['type'] != 'cross':
                continue
            cx, cy = n['pos']
            lens = []
            for (eid, _ang) in n['angles']:
                e = self.edges[eid]
                other = np.array(e.p_v if e.u == nid else e.p_u, dtype=float)
                here  = np.array([cx, cy], dtype=float)
                lens.append(float(np.linalg.norm(other - here)))
            lens.sort()
            L_ref = lens[1] if len(lens) >= 2 else lens[0]
            angs = [a for (_, a) in n['angles']]
            dirs = sorted(((a + math.pi) % math.pi) for a in angs)
            delta = abs(dirs[1] - dirs[0]); delta = min(delta, math.pi - delta)
            angle_factor = 0.80 + 0.20 * min(1.0, delta / (math.pi/2))  # 0.8..1.0
            r_map[nid] = min(0.95 * L_ref, 0.55 * L_ref) * angle_factor
        return r_map
    
    def _poly_is_simple(self, poly, tol=1e-9):
        """
        True iff the closed polyline has no self-intersections (except shared endpoints of adjacent edges).
        poly: list[(x,y)] with the first vertex not repeated at the end.
        """
        m = len(poly)
        if m < 3:
            return False
        P = [np.array(p, float) for p in poly]
        for i in range(m):
            p1 = P[i]
            p2 = P[(i + 1) % m]
            for j in range(i + 1, m):
                # skip adjacent edges (including first-last)
                if j == i or j == (i + 1) % m or (i == 0 and j == m - 1) or ((j + 1) % m == i):
                    continue
                q1 = P[j]
                q2 = P[(j + 1) % m]
                if seg_intersection(p1, p2, q1, q2, tol=tol) is not None:
                    return False
        return True

    def _smoothed_cycle_poly(self, nodes_cycle, r_map, alpha=0.55, steps=48):
        """
        Build world-space polyline for the boundary you SEE:
        straight black segments between crossings, plus cubic Bézier inside each
        smoothed crossing disk. Ports are computed by walking the strand until the
        circle of radius r_map[nid] is hit (same as renderer).
        """
        if not nodes_cycle:
            return []
        n = len(nodes_cycle)

        def append_pt(out, p, eps=1e-10):
            if not out:
                out.append((float(p[0]), float(p[1])))
                return
            x0, y0 = out[-1]
            if abs(p[0]-x0) > eps or abs(p[1]-y0) > eps:
                out.append((float(p[0]), float(p[1])))

        poly = []

        for k in range(n):
            u = nodes_cycle[k]
            v = nodes_cycle[(k + 1) % n]
            node_u = self.nodes[u]
            node_v = self.nodes[v]

            # Start point on (u -> v)
            if node_u['type'] == 'cross' and self.cross_state[u] != 'X':
                Ru = r_map.get(u, 0.0)
                Pstart, _t_out = self._port_on_edge_world(u, v, Ru)
            else:
                Pstart = self.nodes[u]['pos']

            # End point on (u -> v) near v
            if node_v['type'] == 'cross' and self.cross_state[v] != 'X':
                Rv = r_map.get(v, 0.0)
                Pend, t_in = self._port_on_edge_world(v, u, Rv)  # direction from v toward u
            else:
                Pend = self.nodes[v]['pos']
                t_in = None  # not used unless v is a smoothed crossing

            # Put the straight piece endpoint
            if k == 0:
                append_pt(poly, Pstart)
            append_pt(poly, Pend)

            # If v is a smoothed crossing, add the inward Bézier arc (entry -> exit)
            if node_v['type'] == 'cross' and self.cross_state[v] != 'X':
                w = nodes_cycle[(k + 2) % n]  # next node after v
                Rv = r_map.get(v, 0.0)
                Pout, t_out = self._port_on_edge_world(v, w, Rv)

                P0 = np.array(Pend, float)   # entry point on circle
                P3 = np.array(Pout, float)   # exit point on circle
                t_in = np.array(t_in if t_in is not None else (0.0, 0.0), float)
                t_out = np.array(t_out, float)

                # Control points — identical construction to renderer (but world units)
                P1 = P0 - alpha * Rv * t_in
                P2 = P3 - alpha * Rv * t_out

                for tt in np.linspace(0.0, 1.0, steps)[1:-1]:
                    mt = 1.0 - tt
                    Q = (mt**3)*P0 + 3*(mt**2)*tt*P1 + 3*mt*(tt**2)*P2 + (tt**3)*P3
                    append_pt(poly, (float(Q[0]), float(Q[1])))

        # Light simplification: drop tiny steps / almost-colinear middle points
        if len(poly) > 2:
            simp = [poly[0]]
            for i in range(1, len(poly)-1):
                a = np.array(simp[-1], float)
                b = np.array(poly[i], float)
                c = np.array(poly[i+1], float)
                if np.linalg.norm(b - a) < 1e-9:
                    continue
                # area (parallelogram) threshold
                area2 = abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area2 < 1e-12:
                    continue
                simp.append(poly[i])
            simp.append(poly[-1])
            poly = simp

        return poly


# ------------------------------- UI helpers ------------------------------------

class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.hover = False

    def draw(self, surf, font):
        bg = BTN_BG_HOVER if self.hover else BTN_BG
        pygame.draw.rect(surf, bg, self.rect, border_radius=10)
        pygame.draw.rect(surf, BTN_BORDER, self.rect, width=1, border_radius=10)
        txt = font.render(self.label, True, BTN_TEXT)
        tr = txt.get_rect(center=self.rect.center)
        surf.blit(txt, tr)

    def contains(self, pos):
        return self.rect.collidepoint(pos)

# ------------------------------- Renderer --------------------------------------

class PygameRenderer:
    def __init__(self, game: SpliceGame, screen):
        self.game = game
        self.screen = screen
        self.font = pygame.font.SysFont("Arial", 18)
        self.small = pygame.font.SysFont("Arial", 14)
        self.title = pygame.font.SysFont("Arial", 22, bold=True)

        self.layout_ui()

        self._compute_view_transform()

    def _compute_view_transform(self):
        pts = np.array([self.game.nodes[nid]['pos'] for nid in range(len(self.game.nodes))], dtype=float)
        minx, maxx = float(pts[:, 0].min()), float(pts[:, 0].max())
        miny, maxy = float(pts[:, 1].min()), float(pts[:, 1].max())
        # Expand a bit
        dx = maxx - minx
        dy = maxy - miny
        if dx <= 0: dx = 1.0
        if dy <= 0: dy = 1.0
        minx -= 0.08 * dx
        maxx += 0.08 * dx
        miny -= 0.08 * dy
        maxy += 0.08 * dy

        # Fit into screen area (excluding UI bar)
        w, h = self.screen.get_size()
        avail_w, avail_h = w, h - UI_H
        sx = avail_w / (maxx - minx)
        sy = avail_h / (maxy - miny)
        self.scale = min(sx, sy)

        self.port_radius_world = GLOBAL_PORT_RADIUS_PX / self.scale
        self.game.port_radius_world = self.port_radius_world
        # Now that the game knows the true geometric radius, recompute which cycles are Jordan
        self.game._update_seen_cycles()

        # world origin -> screen offset
        # y is inverted (pygame y grows downward)
        world_w = (maxx - minx)
        world_h = (maxy - miny)
        draw_w = self.scale * world_w
        draw_h = self.scale * world_h
        offx = (avail_w - draw_w) * 0.5
        offy = (avail_h - draw_h) * 0.5
        self.minx = minx
        self.maxy = maxy
        self.offx = offx
        self.offy = offy

        # crossing radius in world coords
        self.cross_r_world = 0.045 * max(world_w, world_h)

    def layout_ui(self):
        w, h = self.screen.get_size()
        y0 = max(h - UI_H + 12, 12)  # keep buttons on-screen even if window is short
        btn_w = 170
        btn_h = 44
        pad = 16
        if hasattr(self, "btn_confirm"):
            self.btn_confirm.rect.update(pad, y0, btn_w, btn_h)
            self.btn_new.rect.update(pad*2 + btn_w, y0, btn_w, btn_h)
            self.btn_undo.rect.update(pad*3 + btn_w*2, y0, btn_w, btn_h)
        else:
            self.btn_confirm = Button((pad, y0, btn_w, btn_h), "Confirm")
            self.btn_new = Button((pad*2 + btn_w, y0, btn_w, btn_h), "New")
            self.btn_undo = Button((pad*3 + btn_w*2, y0, btn_w, btn_h), "Undo")


    def world_to_screen(self, p):
        x, y = p
        sx = self.offx + (x - self.minx) * self.scale
        sy = self.offy + (self.maxy - y) * self.scale
        return (int(round(sx)), int(round(sy)))


    def world_to_screen_f(self, p):
        x, y = p
        sx = self.offx + (x - self.minx) * self.scale
        sy = self.offy + (self.maxy - y) * self.scale
        return (sx, sy)  # floats, no rounding


    def screen_to_world(self, p):
        sx, sy = p
        x = (sx - self.offx) / self.scale + self.minx
        y = self.maxy - (sy - self.offy) / self.scale
        return (x, y)
    

    def _next_edge_along_strand(self, edge, at_node):
        """Continue along the same strand through a vertex/crossing."""
        idx_in = self.game.edge_endpoint_stub[(edge.id, at_node)]
        node = self.game.nodes[at_node]
        if node['type'] == 'vertex':         # degree 2
            out_idx = 1 - idx_in
        else:                                # crossing, continue straight across
            out_idx = (idx_in + 2) % 4
        eid2 = node['angles'][out_idx][0]
        e2 = self.game.edges[eid2]
        # return edge oriented to leave 'at_node'
        if e2.u == at_node:
            p0 = np.array(self.game.nodes[at_node]['pos'], float)  # == e2.p_u
            p1 = np.array(e2.p_v, float)
            nxt = e2.v
        else:
            p0 = np.array(self.game.nodes[at_node]['pos'], float)  # == e2.p_v
            p1 = np.array(e2.p_u, float)
            nxt = e2.u
        return e2, p0, p1, nxt

    def _circle_exit_on_stub(self, nid, eid, R_px):
        """First intersection of the strand (starting on (nid,eid)) with the circle
        centered at nid of radius R_px, plus the strand direction there (screen)."""
        Cw = np.array(self.game.nodes[nid]['pos'], float)
        Rw = R_px / self.scale

        # orient the first edge to leave the crossing
        e = self.game.edges[eid]
        if e.u == nid:
            p0 = np.array(e.p_u, float)   # == Cw
            p1 = np.array(e.p_v, float)
            cur_node = e.v
        else:
            p0 = np.array(e.p_v, float)   # == Cw
            p1 = np.array(e.p_u, float)
            cur_node = e.u

        steps = 0
        while steps < 500:
            a = p1 - p0
            A = float(np.dot(a, a))
            b = p0 - Cw
            B = 2.0 * float(np.dot(a, b))
            C = float(np.dot(b, b) - Rw*Rw)
            disc = B*B - 4*A*C

            if disc >= 0.0:
                sq = math.sqrt(disc)
                t1 = (-B - sq) / (2*A)
                t2 = (-B + sq) / (2*A)
                # pick the first intersection forward along the segment
                candidates = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
                # prefer a strictly positive one
                if candidates:
                    t = min([t for t in candidates if t >= 1e-9] or candidates)
                    Pw = p0 + t * a
                    u = a / (np.linalg.norm(a) + 1e-12)
                    upx = np.array([u[0], -u[1]], float)  # flip Y to screen
                    Ppx = self.world_to_screen_f((float(Pw[0]), float(Pw[1])))
                    return (Ppx, upx)

            # this whole segment is inside the circle; move forward along the strand
            e, p0, p1, cur_node = self._next_edge_along_strand(e, cur_node)
            steps += 1

        # very defensive fallback (shouldn’t happen)
        u = (p1 - p0)
        L = float(np.linalg.norm(u)) or 1.0
        u /= L
        upx = np.array([u[0], -u[1]], float)
        Pw = Cw + (Rw * u)
        Ppx = self.world_to_screen_f((float(Pw[0]), float(Pw[1])))
        return (Ppx, upx)
        

    def draw(self, toast_text=None):
        self.screen.fill(BG_COLOR)
        w, h = self.screen.get_size()

        # Draw play area background (optional)
        pygame.draw.rect(self.screen, (245, 246, 249), pygame.Rect(0, 0, w, h - UI_H))

        r = self.cross_r_world

        # --- Per-crossing radius (compute first; used for clipping & smoothing)
        cross_ids = [nid for nid, n in enumerate(self.game.nodes) if n['type'] == 'cross']

        local_r = {}
        lmin_map = {}
        px_floor = 30   # <- minimum smoothing radius in *screen* pixels (bump to taste)
        px_cap   = 48   # <- optional visual cap in pixels (prevents huge bulges)
        r_floor  = px_floor / self.scale
        r_cap    = px_cap   / self.scale

        for nid in cross_ids:
            n = self.game.nodes[nid]
            cx, cy = n['pos']

            # lengths of the 4 incident edge pieces from the crossing
            lens = []
            for (eid, ang) in n['angles']:
                e = self.game.edges[eid]
                other = np.array(e.p_v if e.u == nid else e.p_u, dtype=float)
                here  = np.array([cx, cy], dtype=float)
                L = float(np.linalg.norm(other - here))
                lens.append(L)
            lens.sort()
            L_min = lens[0]
            L_ref = lens[1] if len(lens) >= 2 else L_min   # 2nd-shortest is more stable
            lmin_map[nid] = L_min

            # angle factor (make wider crossings a touch larger)
            angs = [a for (_, a) in n['angles']]
            angs = [((a + math.pi) % (2*math.pi)) - math.pi for a in angs]
            dirs = sorted(((a + math.pi) % math.pi) for a in angs)
            delta = abs(dirs[1] - dirs[0]); delta = min(delta, math.pi - delta)  # acute angle between strands
            angle_factor = 0.80 + 0.20 * min(1.0, delta / (math.pi/2))  # 0.8..1.0

            # candidate radius: at least pixel floor or global base, grow with L_ref
            r_try = max(r_floor, self.cross_r_world, 0.55 * L_ref)
            r_try = min(r_try, r_cap)  # visual cap

            # Use the 2nd-shortest stub (more stable) and enforce the pixel floor
            r_fin = max(min(0.95 * L_ref, r_try), r_floor)
            local_r[nid] = r_fin * angle_factor

        # now inflate once, outside the loop
        pad_world = (max(EDGE_W, SMOOTH_W) * 0.5 + FILL_SAFETY_PX) / self.scale
        r_eff = {nid: local_r[nid] + pad_world for nid in local_r}




        # --- Helpers used below
        def clipped_segment_by_r(p, q, r_u, r_v):
            """Shorten [p,q] by r_u at p-end and r_v at q-end (r_u/r_v may be 0)."""
            p = np.array(p, dtype=float); q = np.array(q, dtype=float)
            v = q - p
            L = np.linalg.norm(v)
            if L < 1e-9 or (r_u + r_v) >= L - 1e-6:
                return None, None
            u = v / L
            a = p + r_u * u
            b = q - r_v * u
            return tuple(a), tuple(b)

        # 1) Draw the entire black polyline (untrimmed)
        for e in self.game.edges:
            A = self.world_to_screen(e.p_u)
            B = self.world_to_screen(e.p_v)
            if A != B:
                pygame.draw.line(self.screen, EDGE_COLOR, A, B, width=EDGE_W)

        # 2) For each crossing: punch a hole at radius R and compute 4 exit ports
        ports = {}  # (nid, stub_idx 0..3) -> (Ppx, u_px)
        for nid in [i for i, n in enumerate(self.game.nodes) if n['type'] == 'cross']:
            cx, cy = self.game.nodes[nid]['pos']
            cpx, cpy = self.world_to_screen((cx, cy))

            # cover the interior with the play-area background to visually trim the black curve
            ERASE_BG = (245, 246, 249)  # same color as the big play-area rect
            # Leave some black under the red so rounding can’t show a seam.
            OVERPAINT_PX = max(EDGE_W, SMOOTH_W)//2 + 2  # ~2px overlap beyond half width
            erase_r = max(1, int(round(GLOBAL_PORT_RADIUS_PX - OVERPAINT_PX)))
            pygame.draw.circle(self.screen, ERASE_BG, (cpx, cpy), erase_r)

            # compute circle exits for the 4 stubs in CCW order
            for s_idx, (eid, _ang) in enumerate(self.game.nodes[nid]['angles']):
                Ppx, upx = self._circle_exit_on_stub(nid, eid, GLOBAL_PORT_RADIUS_PX)
                ports[(nid, s_idx)] = (np.array(Ppx, float), np.array(upx, float))

        # 3) Draw the red smoothing between those ports
        def draw_bezier_px(P0, P3, u_in_px, u_out_px, R_px, alpha=0.55, steps=28):
            P0 = np.array(P0, float); P3 = np.array(P3, float)
            P1 = P0 - alpha * R_px * u_in_px
            P2 = P3 - alpha * R_px * u_out_px
            pts = []
            for t in np.linspace(0.0, 1.0, steps):
                mt = 1.0 - t
                Q = (mt**3)*P0 + 3*(mt**2)*t*P1 + 3*mt*(t**2)*P2 + (t**3)*P3
                pts.append((int(round(Q[0])), int(round(Q[1]))))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, SMOOTH_COLOR, False, pts, width=SMOOTH_W)

        
        def draw_cubic_dir(A, B, tA, tB, scale_px, alpha=0.55, steps=28):
            """Cubic from A to B with tangent tA at A and tB at B (unit dirs in screen space)."""
            A = np.array(A, float); B = np.array(B, float)
            tA = np.array(tA, float); tB = np.array(tB, float)
            P1 = A + alpha * scale_px * tA
            P2 = B - alpha * scale_px * tB
            pts = []
            for tt in np.linspace(0.0, 1.0, steps):
                mt = 1.0 - tt
                Q = (mt**3)*A + 3*(mt**2)*tt*P1 + 3*mt*(tt**2)*P2 + (tt**3)*B
                pts.append((int(round(Q[0])), int(round(Q[1]))))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, SMOOTH_COLOR, False, pts, width=SMOOTH_W)



        for nid in [i for i, n in enumerate(self.game.nodes) if n['type'] == 'cross']:
            inc_sorted = self.game.nodes[nid]['angles']  # CCW, len==4
            # use pending state if present (preview)
            state = self.game.cross_state[nid]
            if self.game.pending_cross is not None and self.game.pending_cross[0] == nid:
                state = self.game.pending_cross[1]

            # gather port positions/tangents in CCW order
            ends = [ports[(nid, s_idx)] for s_idx in range(4)]

            if state == 'X':
                # ensure the X passes exactly through the crossing center
                Cpx = np.array(self.world_to_screen_f(self.game.nodes[nid]['pos']), float)

                # Opposite pairs (0↔2) and (1↔3)
                for i, j in [(0, 2), (1, 3)]:
                    P0, u0 = ends[i]   # port position & outward screen-space tangent along the strand
                    P3, u3 = ends[j]

                    # Port -> Center: start tangent points inward (-u0), end tangent at center points back towards the port (+u0)
                    draw_cubic_dir(P0, Cpx, -u0, +u0, GLOBAL_PORT_RADIUS_PX)

                    # Center -> Opposite port: start tangent towards the port (+u3), end tangent at port points inward (-u3)
                    draw_cubic_dir(Cpx, P3, +u3, -u3, GLOBAL_PORT_RADIUS_PX)
            elif state == 'A':
                for i, j in [(0, 1), (2, 3)]:
                    P0, u_in  = ends[i]
                    P3, u_out = ends[j]
                    draw_bezier_px(P0, P3, u_in, u_out, GLOBAL_PORT_RADIUS_PX)
            elif state == 'B':
                for i, j in [(1, 2), (3, 0)]:
                    P0, u_in  = ends[i]
                    P3, u_out = ends[j]
                    draw_bezier_px(P0, P3, u_in, u_out, GLOBAL_PORT_RADIUS_PX)




        # --- Draw claimed regions (fills) using the *smoothed* curve geometry
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        def build_smoothed_cycle_world_path(nodes_cycle, r_map, alpha=0.55, steps=64):

            """
            Build a closed *world-space* path for a cycle using clipped lines + inward Bézier
            at smoothed crossings. Robust to links that are shorter than r_u + r_v.
            nodes_cycle: list of node ids (WITHOUT repeating the first at the end).
            """
            if not nodes_cycle:
                return []
            n = len(nodes_cycle)

            def unit_vec(a, b):
                a = np.array(a, dtype=float)
                b = np.array(b, dtype=float)
                v = b - a
                L = np.linalg.norm(v)
                if L < 1e-12:
                    return np.array([0.0, 0.0], dtype=float)
                return v / L

            # Precompute clipped endpoints for each directed link k: k -> (k+1)
            a_pts = [None] * n  # start near node k
            b_pts = [None] * n  # end near node k+1

            for k in range(n):
                u_id = nodes_cycle[k]
                v_id = nodes_cycle[(k + 1) % n]
                Pu = np.array(self.game.nodes[u_id]['pos'], dtype=float)
                Pv = np.array(self.game.nodes[v_id]['pos'], dtype=float)

                r_u = r_map.get(u_id, 0.0) if self.game.nodes[u_id]['type'] == 'cross' else 0.0
                r_v = r_map.get(v_id, 0.0) if self.game.nodes[v_id]['type'] == 'cross' else 0.0

                # Try the geometric clip
                Au, Bv = clipped_segment_by_r(Pu, Pv, r_u, r_v)
                if Au is None:
                    # Degenerate link (shorter than r_u + r_v). Synthesize stable endpoints:
                    d = unit_vec(Pu, Pv)
                    Au = Pu + r_u * d
                    Bv = Pv - r_v * d
                    # If they still overlap, collapse to the midpoint to avoid aborting the fill
                    if np.linalg.norm(np.array(Bv) - np.array(Au)) < 1e-9:
                        M = (np.array(Au) + np.array(Bv)) * 0.5
                        Au = (float(M[0]), float(M[1]))
                        Bv = (float(M[0]), float(M[1]))

                a_pts[k] = np.array(Au, dtype=float)
                b_pts[k] = np.array(Bv, dtype=float)

            # Assemble the path: a0, (line to b0), (arc at node0 from b0->a1), (line to b1), ...
            path_world = []
            path_world.append((float(a_pts[0][0]), float(a_pts[0][1])))

            for k in range(n):
                # 1) straight piece end
                path_world.append((float(b_pts[k][0]), float(b_pts[k][1])))

                # 2) if next node is a smoothed crossing, add inward Bézier from b_k -> a_{k+1}
                nid = nodes_cycle[(k + 1) % n]
                node = self.game.nodes[nid]
                if node['type'] == 'cross':
                    state = self.game.cross_state[nid]
                    if state != 'X':
                        prev_id = nodes_cycle[k]
                        next_id = nodes_cycle[(k + 2) % n]
                        C = np.array(node['pos'], dtype=float)
                        u_in  = unit_vec(C, self.game.nodes[prev_id]['pos'])  # along incoming stub
                        u_out = unit_vec(C, self.game.nodes[next_id]['pos'])  # along outgoing stub
                        r_loc = r_map.get(nid, 0.0)

                        P0 = b_pts[k]
                        P3 = a_pts[(k + 1) % n]
                        # Pull control points back *toward* the crossing so the curve bends inward
                        P1 = P0 - alpha * r_loc * u_in
                        P2 = P3 - alpha * r_loc * u_out

                        # Sample cubic Bézier; skip endpoints (we already added b_k, and a_{k+1} arrives later)
                        for t in np.linspace(0.0, 1.0, steps)[1:-1]:
                            mt = 1.0 - t
                            Q = (mt**3)*P0 + 3*(mt**2)*t*P1 + 3*mt*(t**2)*P2 + (t**3)*P3
                            path_world.append((float(Q[0]), float(Q[1])))

            return path_world

        for cl in self.game.claimed:
            pts_world = cl.get('poly')  # world-space polygon stored at scoring time
            if pts_world and len(pts_world) >= 3:
                pts_screen = [self.world_to_screen(p) for p in pts_world]
                color = CLAIM_FILL_COLOR_P1 if cl['owner'] == 0 else CLAIM_FILL_COLOR_P2
                pygame.draw.polygon(overlay, color, pts_screen)

        self.screen.blit(overlay, (0, 0))



        # --- Draw crossings (dots + pending halo)
        for nid in cross_ids:
            x, y = self.game.nodes[nid]['pos']
            px, py = self.world_to_screen((x, y))

            pending = (self.game.pending_cross is not None and self.game.pending_cross[0] == nid)
            state = self.game.cross_state[nid]
            eff_state = state
            if pending:  # show the *pending* state in the UI
                eff_state = self.game.pending_cross[1]

            if pending and eff_state in ('A', 'B'):
                # Big halo so the user clearly sees the proposed change; no black dot.
                base_r_px = r_eff.get(nid, self.cross_r_world) * self.scale
                halo_r_px = int(round(max(22, min(90, base_r_px * 1.25))))  # tweak min/max/scale to taste
                pygame.draw.circle(self.screen, PENDING_HALO, (px, py), halo_r_px, width=3)
                continue  # do not draw the black dot

            # For X (or pending back to X): small halo + black dot
            if eff_state == 'X':
                if pending:
                    pygame.draw.circle(self.screen, PENDING_HALO, (px, py), 9, width=2)
                pygame.draw.circle(self.screen, CROSS_DOT_COLOR, (px, py), 4)
            # If eff_state is A/B and not pending (i.e., already confirmed), draw nothing here.

        # --- Draw labels on claimed regions (centroid of *smoothed* polygon)
        for cl in self.game.claimed:
            pts_world = cl.get('poly')
            if pts_world and len(pts_world) >= 3:
                # ensure saved label is inside with margin; recompute if necessary
                desired_margin_px = 12.0
                if self.game.port_radius_world is not None:
                    margin_world = (desired_margin_px / float(GLOBAL_PORT_RADIUS_PX)) * float(self.game.port_radius_world)
                else:
                    margin_world = 0.5

                L = cl.get('label', None)
                if (L is None) or (not point_in_poly(L, pts_world)) or \
                (self.game._distance_to_edges(L, pts_world) < margin_world):
                    cl['label'] = self.game._best_label_point(pts_world, margin_world=margin_world)

                cx, cy = cl['label']
                sx, sy = self.world_to_screen((cx, cy))
                tag = f"P{cl['owner']+1}" + ("★" if cl['outside'] else "")
                txt = self.small.render(tag, True, (0, 0, 0))
                tr = txt.get_rect(center=(sx, sy))
                self.screen.blit(txt, tr)

        # --- Title & score
        title_txt = self.title.render("Splice — click a crossing (X→A→B→X), then Confirm", True, TEXT_COLOR)
        self.screen.blit(title_txt, (16, 10))
        score = f"P1: {self.game.scores[0]}    P2: {self.game.scores[1]}    Turn: P{self.game.player+1}"
        score_txt = self.font.render(score, True, TEXT_COLOR)
        self.screen.blit(score_txt, (16, 42))

        if toast_text:
            toast = self.font.render(toast_text, True, (0, 120, 40))
            self.screen.blit(toast, (16, 66))

        # --- UI buttons bar
        pygame.draw.rect(self.screen, (240, 241, 245), pygame.Rect(0, h - UI_H, w, UI_H))
        mouse_pos = pygame.mouse.get_pos()
        for btn in (self.btn_confirm, self.btn_new, self.btn_undo):
            btn.hover = btn.contains(mouse_pos)
            btn.draw(self.screen, self.font)

    def buttons(self):
        return [self.btn_confirm, self.btn_new, self.btn_undo]

# --------------------------------- App -----------------------------------------

class App:
    def __init__(self, seed=None, min_cross_sep=0.6, min_cross_angle_deg=25.0):
        pygame.init()
        pygame.display.set_caption("Splice (pygame)")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.game = SpliceGame(seed=seed, min_cross_sep=min_cross_sep,
                               min_cross_angle_deg=min_cross_angle_deg)
        self.renderer = PygameRenderer(self.game, self.screen)
        self.undo_stack = []  # snapshots

    def snapshot(self):
        cs = dict(self.game.cross_state)
        player = self.game.player
        scores = list(self.game.scores)
        claimed = [{'nodes': list(c['nodes']), 'poly': list(c['poly']), 'label': tuple(c['label']),
                    'owner': c['owner'], 'outside': c['outside']} for c in self.game.claimed]
        seen = set(self.game.seen_simple_cycles)
        return (cs, player, scores, claimed, seen)

    def restore(self, snap):
        cs, player, scores, claimed, seen = snap
        self.game.cross_state = dict(cs)
        self.game.player = player
        self.game.scores = list(scores)
        self.game.claimed = [{'nodes': list(c['nodes']), 'poly': list(c['poly']), 'label': tuple(c['label']),
                              'owner': c['owner'], 'outside': c['outside']} for c in claimed]
        self.game.seen_simple_cycles = set(seen)
        self.game.pending_cross = None

    async def run(self):
        toast = None
        toast_timer = 0
        running = True
        while running:
            dt = self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_RETURN, pygame.K_c):
                        # Confirm move
                        if self.game.pending_cross is not None:
                            self.undo_stack.append(self.snapshot())
                            summary = self.game.commit_and_score()
                            awards = summary.get('awards', [])
                            if awards:
                                msg = "; ".join("+1 " + ("outside" if a['type']=='outside' else "disk") + f" (P{a['player']+1})" for a in awards)

                                toast = msg
                                toast_timer = 1500
                    elif event.key == pygame.K_n:
                        # New random curve
                        try:
                            self.game = SpliceGame(min_cross_sep=self.game.min_cross_sep,
                                                min_cross_angle_deg=math.degrees(self.game.min_cross_angle_rad))
                            self.renderer = PygameRenderer(self.game, self.screen)
                            self.undo_stack.clear()
                        except RuntimeError as e:
                            toast = str(e)
                            toast_timer = 2500
                    elif event.key in (pygame.K_u, pygame.K_BACKSPACE):
                        # Undo
                        if self.undo_stack:
                            snap = self.undo_stack.pop()
                            self.restore(snap)

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    # Check buttons
                    if self.renderer.btn_confirm.contains((mx, my)):
                        if self.game.pending_cross is not None:
                            self.undo_stack.append(self.snapshot())
                            summary = self.game.commit_and_score()
                            awards = summary.get('awards', [])
                            if awards:
                                msg = "; ".join("+1 " + ("outside" if a['type']=='outside' else "disk") + f" (P{a['player']+1})" for a in awards)
                                toast = msg
                                toast_timer = 1500  # ms
                        # else: no-op
                    elif self.renderer.btn_new.contains((mx, my)):
                        try:
                            self.game = SpliceGame(min_cross_sep=self.game.min_cross_sep,
                                                   min_cross_angle_deg=math.degrees(self.game.min_cross_angle_rad))
                            self.renderer = PygameRenderer(self.game, self.screen)
                            self.undo_stack.clear()
                        except RuntimeError as e:
                            toast = str(e)
                            toast_timer = 2500
                    elif self.renderer.btn_undo.contains((mx, my)):
                        if self.undo_stack:
                            snap = self.undo_stack.pop()
                            self.restore(snap)
                    else:
                        # Board click — cycle a crossing
                        nid = self.game.find_nearest_crossing(mx, my, self.renderer.world_to_screen, max_pix=28)
                        if nid is None:
                            pass
                        else:
                            # Ignore clicks on already-changed crossings (not 'X')
                            if self.game.cross_state[nid] != 'X':
                                pass
                            else:
                                # (keep the existing pending logic below unchanged)
                                if self.game.pending_cross is not None:
                                    nid_pending, pstate = self.game.pending_cross
                                    if nid != nid_pending and pstate != self.game.cross_state[nid_pending]:
                                        pass
                                    else:
                                        self.game.cycle_crossing_state(nid_pending if nid is None else nid)
                                else:
                                    self.game.cycle_crossing_state(nid)
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self.renderer.screen = self.screen
                    self.renderer._compute_view_transform()
                    self.renderer.layout_ui()

            if toast_timer > 0:
                toast_timer -= dt
                if toast_timer <= 0:
                    toast = None

            self.renderer.draw(toast_text=toast)
            pygame.display.flip()

            await asyncio.sleep(0)
        pygame.quit()

# --------------------------------- Main ----------------------------------------

async def main():
    seed = None
    if len(sys.argv) > 1:
        try:
            seed = int(sys.argv[1])
        except Exception:
            seed = None
    app = App(seed=seed, min_cross_sep=1.5, min_cross_angle_deg=25.0)
    await app.run()

if __name__ == "__main__":
    asyncio.run(main())