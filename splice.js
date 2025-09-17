// ============================ Splice (Web Canvas) ============================
// Responsive canvas + robust generator (backoff) + faithful logic/rendering
// ============================================================================

(() => {
  // ------------------------------ Config ------------------------------------
  const EDGE_COLOR = "#1e1e1e";
  const SMOOTH_COLOR = "#dc1e1e";
  const CROSS_DOT_COLOR = "#141414";
  const CLAIM_FILL_COLOR_P1 = "rgba(80, 140, 255, 0.5)";
  const CLAIM_FILL_COLOR_P2 = "rgba(255, 140, 80, 0.5)";
  const PENDING_HALO = "#1478ff";
  const PLAY_BG = "#f5f6f9";

  const rootStyle = document.documentElement && document.documentElement.style;
  if (rootStyle){
    rootStyle.setProperty("--player1-bg", CLAIM_FILL_COLOR_P1);
    rootStyle.setProperty("--player2-bg", CLAIM_FILL_COLOR_P2);
  }

  const EDGE_W = 2;
  const SMOOTH_W = EDGE_W;
  const FILL_SAFETY_PX = 3.0;
  const GLOBAL_PORT_RADIUS_PX = 24;

  // --------------------------- Utilities ------------------------------------

  // Seeded RNG (Mulberry32)
  function mulberry32(seed) {
    let t = seed >>> 0;
    return function() {
      t += 0x6D2B79F5;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }
  let RAND = Math.random;
  function set_seed(seed=null){
    if (seed == null) seed = Math.floor(Math.random() * (1<<30));
    RAND = mulberry32(seed >>> 0);
    return seed;
  }

  const vsub=(a,b)=>[a[0]-b[0], a[1]-b[1]];
  const vadd=(a,b)=>[a[0]+b[0], a[1]+b[1]];
  const vmul=(a,s)=>[a[0]*s,a[1]*s];
  const vlen=a=>Math.hypot(a[0],a[1]);
  const vnorm=a=>{ const L=vlen(a)||1e-12; return [a[0]/L,a[1]/L]; };

  function angOf(vec){ return Math.atan2(vec[1], vec[0]); }
  function dot(ax,ay,bx,by){ return ax*bx+ay*by; }

  function shoelace_area(poly){
    const n=poly.length; if (n<3) return 0;
    let s=0;
    for(let i=0;i<n;i++){
      const [x1,y1]=poly[i], [x2,y2]=poly[(i+1)%n];
      s += x1*y2 - x2*y1;
    }
    return 0.5*s;
  }
  function point_in_poly(pt, poly){
    const [x,y]=pt;
    let inside=false; const n=poly.length; if (n<3) return false;
    for(let i=0;i<n;i++){
      const [x1,y1]=poly[i], [x2,y2]=poly[(i+1)%n];
      const cond=((y1>y)!==(y2>y));
      if (cond){
        const xinters=(x2-x1)*(y-y1)/((y2-y1)||1e-16)+x1;
        if (xinters>x) inside=!inside;
      }
    }
    return inside;
  }
  function seg_intersection(p, p2, q, q2, tol=1e-9){
    const rx = p2[0]-p[0], ry=p2[1]-p[1];
    const sx = q2[0]-q[0], sy=q2[1]-q[1];
    const rxs = rx*sy - ry*sx;
    const qmpx = q[0]-p[0], qmpy = q[1]-p[1];
    if (Math.abs(rxs)<tol) return null;
    const t = (qmpx*sy - qmpy*sx)/rxs;
    const u = (qmpx*ry - qmpy*rx)/rxs;
    if (tol < t && t < 1-tol && tol < u && u < 1-tol){
      return [t,u, [p[0]+t*rx, p[1]+t*ry]];
    }
    return null;
  }
  function polygon_centroid(poly){
    const A=shoelace_area(poly);
    if (Math.abs(A)<1e-12){
      let sx=0, sy=0; for (const [x,y] of poly){ sx+=x; sy+=y; }
      return [sx/poly.length, sy/poly.length];
    }
    let cx=0, cy=0;
    const n=poly.length;
    for(let i=0;i<n;i++){
      const [x1,y1]=poly[i], [x2,y2]=poly[(i+1)%n];
      const cross = x1*y2 - x2*y1;
      cx += (x1+x2)*cross;
      cy += (y1+y2)*cross;
    }
    return [cx/(6*A), cy/(6*A)];
  }
  function interior_point(poly, tries=200){
    if (!poly || !poly.length) return [0,0];
    let minx=Infinity, miny=Infinity, maxx=-Infinity, maxy=-Infinity;
    for (const [x,y] of poly){ if (x<minx) minx=x; if (x>maxx) maxx=x; if (y<miny) miny=y; if (y>maxy) maxy=y; }
    for(let i=0;i<tries;i++){
      const x = minx + RAND()*(maxx-minx);
      const y = miny + RAND()*(maxy-miny);
      if (point_in_poly([x,y], poly)) return [x,y];
    }
    return polygon_centroid(poly);
  }

  // ------------------ Random curve & immersed graph --------------------------

  function random_fourier_closed_polyline(M=140, modes=4, radius=5.0){
    function coeffs(){
      const a=new Array(modes+1).fill(0);
      const b=new Array(modes+1).fill(0);
      for(let k=1;k<=modes;k++){
        // normal-ish via sum of uniforms
        const ga = (RAND()+RAND()+RAND()+RAND()-2)*0.8;
        const gb = (RAND()+RAND()+RAND()+RAND()-2)*0.8;
        a[k]=ga / Math.pow(k,1.3);
        b[k]=gb / Math.pow(k,1.3);
      }
      return [a,b];
    }
    const t = new Array(M); for(let i=0;i<M;i++) t[i]=2*Math.PI*i/M;
    const [ax,bx]=coeffs(), [ay,by]=coeffs();

    const x = new Array(M).fill(0), y=new Array(M).fill(0);
    for(let k=1;k<=modes;k++){
      for(let i=0;i<M;i++){
        x[i]+= ax[k]*Math.cos(k*t[i]) + bx[k]*Math.sin(k*t[i]);
        y[i]+= ay[k]*Math.cos(k*t[i]) + by[k]*Math.sin(k*t[i]);
      }
    }
    const mean = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
    const sx=mean(x), sy=mean(y);
    let vx=0, vy=0; for(let i=0;i<M;i++){ vx+=(x[i]-sx)**2; vy+=(y[i]-sy)**2; }
    const stdx=Math.sqrt(vx/M)+1e-9, stdy=Math.sqrt(vy/M)+1e-9;
    const pts=new Array(M);
    for(let i=0;i<M;i++) pts[i]=[ radius*(x[i]-sx)/stdx, radius*(y[i]-sy)/stdy ];
    return pts;
  }

  function build_immersed_graph(poly, sep_min=null, angle_min_rad=null){
    const N=poly.length;
    const segs=[]; for(let i=0;i<N;i++) segs.push([i, [poly[i], poly[(i+1)%N]]]);

    const seg_to_splits=new Map();
    const nodes=[];
    const node_id_of_point=new Map();
    const tol=1e-7;
    const key_of_pt = pt => `${Math.round(pt[0]/tol)*tol},${Math.round(pt[1]/tol)*tol}`;

    for(let i=0;i<N;i++){
      const p=[+poly[i][0], +poly[i][1]];
      const nid=nodes.length;
      nodes.push({pos:p, type:'vertex', inc:[], angles:[]});
      node_id_of_point.set(`v:${i}`, nid);
    }

    for (let i=0;i<N;i++){
      const [a,b]=segs[i][1];
      for (let j=i+1;j<N;j++){
        if (j===i || j===(i+1)%N || i===(j+1)%N) continue;
        const [c,d]=segs[j][1];
        const res=seg_intersection(a,b,c,d);
        if (!res) continue;
        const [t,u,pt]=res;
        if (!seg_to_splits.has(i)) seg_to_splits.set(i,[]);
        if (!seg_to_splits.has(j)) seg_to_splits.set(j,[]);
        seg_to_splits.get(i).push([t,pt]);
        seg_to_splits.get(j).push([u,pt]);
        const k=key_of_pt(pt);
        if (!node_id_of_point.has(`x:${k}`)){
          const nid=nodes.length;
          nodes.push({pos:[+pt[0], +pt[1]], type:'cross', inc:[], angles:[]});
          node_id_of_point.set(`x:${k}`, nid);
        }
      }
    }

    const edges=[];
    function node_id_for_seg_endpoint(seg_idx, is_end){
      const vid = !is_end ? seg_idx : ((seg_idx+1)%N);
      return node_id_of_point.get(`v:${vid}`);
    }
    for (let i=0;i<N;i++){
      const a=segs[i][1][0], b=segs[i][1][1];
      const splits = seg_to_splits.get(i) || [];
      const enriched = [[0.0, a.slice(), node_id_for_seg_endpoint(i,false)]]
        .concat(splits.map(([t,pt]) => [t, pt.slice(), null]))
        .concat([[1.0, b.slice(), node_id_for_seg_endpoint(i,true)]]);
      enriched.sort((A,B)=>A[0]-B[0]);
      for (let k=0;k<enriched.length-1;k++){
        let [t1,p1,nid1]=enriched[k];
        let [t2,p2,nid2]=enriched[k+1];
        if (nid1==null){ nid1=node_id_of_point.get(`x:${key_of_pt(p1)}`); }
        if (nid2==null){ nid2=node_id_of_point.get(`x:${key_of_pt(p2)}`); }
        const eid=edges.length;
        edges.push({id:eid, u:nid1, v:nid2, p_u:[+p1[0],+p1[1]], p_v:[+p2[0],+p2[1]]});
        nodes[nid1].inc.push(eid);
        nodes[nid2].inc.push(eid);
      }
    }

    // degree-4 at crosses
    for (let i=0;i<nodes.length;i++){
      const n=nodes[i];
      if (n.type==='cross' && n.inc.length!==4) return null;
    }

    // Angles CCW
    for (let nid=0;nid<nodes.length;nid++){
      const n=nodes[nid]; n.angles=[];
      for (const eid of n.inc){
        const e=edges[eid];
        const other = (e.u===nid) ? e.p_v : e.p_u;
        const here  = (e.u===nid) ? e.p_u : e.p_v;
        const vec = vsub(other, here);
        n.angles.push([eid, angOf(vec)]);
      }
      n.angles.sort((A,B)=>A[1]-B[1]);
    }

    // Constraints
    if (sep_min!=null || angle_min_rad!=null){
      const cross_ids = [];
      for (let i=0;i<nodes.length;i++) if (nodes[i].type==='cross') cross_ids.push(i);
      if (cross_ids.length){
        if (sep_min!=null){
          let dmin=Infinity;
          for (let i=0;i<cross_ids.length;i++){
            const p=nodes[cross_ids[i]].pos;
            for (let j=i+1;j<cross_ids.length;j++){
              const q=nodes[cross_ids[j]].pos;
              const d=Math.hypot(p[0]-q[0], p[1]-q[1]);
              if (d<dmin) dmin=d;
            }
          }
          if (dmin < +sep_min) return null;
        }
        if (angle_min_rad!=null){
          for (const nid of cross_ids){
            let angs = nodes[nid].angles.map(a=>a[1]).map(a=>((a+Math.PI)%(2*Math.PI))-Math.PI);
            let dirs=angs.map(a=>((a+Math.PI)%Math.PI)).sort((a,b)=>a-b);
            const uniq=[]; for (const d of dirs){ if (!uniq.some(u=>Math.abs(u-d)<1e-4)) uniq.push(d); }
            if (uniq.length!==2) return null;
            let delta=Math.abs(uniq[0]-uniq[1]); delta=Math.min(delta, Math.PI-delta);
            if (delta < +angle_min_rad) return null;
          }
        }
      }
    }
    return [nodes, edges];
  }

  // --------------------------- Core game model --------------------------------

  class SpliceGame {
    constructor(seed=null, min_cross_sep=0.6, min_cross_angle_deg=25.0){
      this.min_cross_sep = +min_cross_sep;
      this.min_cross_angle_rad = (Math.PI/180)*(+min_cross_angle_deg);
      this.seed = set_seed(seed);
      const [nodes,edges] = this._make_good_random_graph();
      this.nodes = nodes; this.edges=edges;
      this._build_halfedges();
      this.cross_state = {};
      for (let nid=0;nid<this.nodes.length;nid++) if (this.nodes[nid].type==='cross') this.cross_state[nid]='X';
      this.pending_cross=null;
      this.player=0;
      this.scores=[0,0];
      this.claimed=[];
      this.seen_simple_cycles=new Set();
      this._update_seen_cycles();
      this.port_radius_world = null; // renderer will set
    }

    _make_good_random_graph(){
      for (let tries=0; tries<256; tries++){
        const r = Math.max(5.0, 2.0*this.min_cross_sep);
        const poly = random_fourier_closed_polyline(140,4,r);
        const res = build_immersed_graph(poly, this.min_cross_sep, this.min_cross_angle_rad);
        if (res){
          const [nodes, edges] = res;
          const n_cross = nodes.reduce((a,n)=>a+(n.type==='cross'?1:0),0);
          if (n_cross>=6) return [nodes, edges];
        }
      }
      throw new Error("Could not generate a curve meeting spacing/angle constraints; loosen parameters or retry.");
    }

    _build_halfedges(){
      this.stub_to_he = Array.from({length:this.nodes.length},()=>({}));
      this.halfedges=[];
      this.edge_to_he={};
      for (const e of this.edges){
        const h1={eid:e.id, tail:e.u, head:e.v};
        const h2={eid:e.id, tail:e.v, head:e.u};
        const h1_id=this.halfedges.length; this.halfedges.push(h1);
        const h2_id=this.halfedges.length; this.halfedges.push(h2);
        this.edge_to_he[`${e.id},0`]=h1_id;
        this.edge_to_he[`${e.id},1`]=h2_id;
      }
      this.edge_endpoint_stub={};
      for (let nid=0;nid<this.nodes.length;nid++){
        const inc_sorted=this.nodes[nid].angles;
        for (let idx=0; idx<inc_sorted.length; idx++){
          const [eid,_ang]=inc_sorted[idx];
          const e=this.edges[eid];
          const he_id = (e.u===nid) ? this.edge_to_he[`${eid},0`] : this.edge_to_he[`${eid},1`];
          this.stub_to_he[nid][idx]=he_id;
        }
      }
      for (let nid=0;nid<this.nodes.length;nid++){
        for (let idx=0; idx<this.nodes[nid].angles.length; idx++){
          const [eid,_]=this.nodes[nid].angles[idx];
          this.edge_endpoint_stub[`${eid},${nid}`]=idx;
        }
      }
    }

    _next_edge_along_strand(edge, at_node){
      const idx_in = this.edge_endpoint_stub[`${edge.id},${at_node}`];
      const node = this.nodes[at_node];
      const out_idx = (node.type==='vertex') ? (1-idx_in) : ((idx_in+2)%4);
      const eid2 = node.angles[out_idx][0];
      const e2=this.edges[eid2];
      let p0,p1,nxt;
      if (e2.u===at_node){ p0=this.nodes[at_node].pos; p1=e2.p_v; nxt=e2.v; }
      else{ p0=this.nodes[at_node].pos; p1=e2.p_u; nxt=e2.u; }
      return [e2, [p0[0],p0[1]], [p1[0],p1[1]], nxt];
    }

    _point_seg_dist(p,a,b){
      const px=p[0], py=p[1], ax=a[0], ay=a[1], bx=b[0], by=b[1];
      const vx=bx-ax, vy=by-ay, wx=px-ax, wy=py-ay;
      const vv=vx*vx+vy*vy; if (vv<=1e-20) return Math.hypot(px-ax, py-ay);
      const t=Math.max(0, Math.min(1, (wx*vx+wy*vy)/vv));
      const cx=ax+t*vx, cy=ay+t*vy;
      return Math.hypot(px-cx, py-cy);
    }
    _distance_to_edges(p, poly){
      let mind=Infinity;
      for (let i=0;i<poly.length;i++){
        const a=poly[i], b=poly[(i+1)%poly.length];
        const d=this._point_seg_dist(p,a,b);
        if (d<mind) mind=d;
      }
      return mind;
    }
    _best_label_point(poly, margin_world=null, grid=12, refine=[0.5,0.25,0.12]){
      if (!poly || !poly.length) return [0,0];
      let minx=Infinity,miny=Infinity,maxx=-Infinity,maxy=-Infinity;
      for(const [x,y] of poly){ if (x<minx) minx=x; if (x>maxx) maxx=x; if (y<miny) miny=y; if (y>maxy) maxy=y; }
      let best_pt=null, best_d=-1.0;
      const c0=polygon_centroid(poly);
      if (point_in_poly(c0, poly)){ const d0=this._distance_to_edges(c0, poly); best_pt=c0; best_d=d0; }
      for (let i=0;i<grid;i++){
        for (let j=0;j<grid;j++){
          const x=minx+(i+0.5)*(maxx-minx)/grid, y=miny+(j+0.5)*(maxy-miny)/grid;
          const pt=[x,y];
          if (point_in_poly(pt, poly)){
            const d=this._distance_to_edges(pt, poly);
            if (d>best_d){ best_pt=pt; best_d=d; }
          }
        }
      }
      if (best_pt && best_d>0){
        for (const frac of refine){
          const radius=best_d*frac;
          for (let k=0;k<20;k++){
            const ang=2*Math.PI*(k/20 + RAND()*0.02);
            const pt=[best_pt[0]+radius*Math.cos(ang), best_pt[1]+radius*Math.sin(ang)];
            if (point_in_poly(pt, poly)){
              const d=this._distance_to_edges(pt, poly);
              if (d>best_d){ best_pt=pt; best_d=d; }
            }
          }
        }
      }
      if (!best_pt){ best_pt=interior_point(poly); }
      return [best_pt[0], best_pt[1]];
    }

    _port_on_edge_world(cross_id, neighbor_id, Rw, max_steps=400){
      const Cw=this.nodes[cross_id].pos.slice();
      let eid0=null;
      for (const eid of this.nodes[cross_id].inc){
        const e=this.edges[eid];
        if (e.u===neighbor_id || e.v===neighbor_id){ eid0=eid; break; }
      }
      if (eid0==null) return [[Cw[0]+Rw, Cw[1]],[1,0]];
      let e=this.edges[eid0], p0,p1,cur_node;
      if (e.u===cross_id){ p0=e.p_u.slice(); p1=e.p_v.slice(); cur_node=e.v; }
      else{ p0=e.p_v.slice(); p1=e.p_u.slice(); cur_node=e.u; }

      let steps=0;
      while (steps<max_steps){
        const a=vsub(p1,p0);
        const A=dot(a[0],a[1],a[0],a[1]);
        const b=vsub(p0,Cw);
        const B=2*dot(a[0],a[1],b[0],b[1]);
        const C=dot(b[0],b[1],b[0],b[1]) - Rw*Rw;
        const disc=B*B-4*A*C;
        if (disc>=0 && A>1e-20){
          const sq=Math.sqrt(disc);
          const t1=(-B - sq)/(2*A), t2=(-B + sq)/(2*A);
          const cands=[t1,t2].filter(t=>0<=t && t<=1);
          if (cands.length){
            const pos=cands.filter(t=>t>=1e-9);
            const t = pos.length? Math.min(...pos) : Math.min(...cands);
            const Pw=vadd(p0, vmul(a,t));
            const u=vnorm(a);
            return [[Pw[0],Pw[1]], [u[0],u[1]]];
          }
        }
        [e,p0,p1,cur_node]=this._next_edge_along_strand(e, cur_node);
        steps++;
      }
      let u=vnorm(vsub(p1,p0));
      const Pw=vadd(Cw, vmul(u,Rw));
      return [[Pw[0],Pw[1]], [u[0],u[1]]];
    }

    static _pairing_for_cross(deg, state){
      if (state==='X') return {0:2,2:0,1:3,3:1};
      if (state==='A') return {0:1,1:0,2:3,3:2};
      if (state==='B') return {1:2,2:1,3:0,0:3};
      throw new Error("Unknown state");
    }
    static _pairing_for_vertex(){ return {0:1,1:0}; }
    _node_pairing(nid, use_pending=false){
      const n=this.nodes[nid];
      if (n.type==='vertex') return SpliceGame._pairing_for_vertex(2);
      let state=this.cross_state[nid];
      if (use_pending && this.pending_cross && this.pending_cross[0]===nid) state=this.pending_cross[1];
      return SpliceGame._pairing_for_cross(4, state);
    }
    successor(he_id, use_pending=false){
      const h=this.halfedges[he_id];
      const head=h.head, eid=h.eid;
      const s_in=this.edge_endpoint_stub[`${eid},${head}`];
      const pairing=this._node_pairing(head, use_pending);
      const s_out=pairing[s_in];
      return this.stub_to_he[head][s_out];
    }
    enumerate_cycles(use_pending=false){
      const visited=new Set(), cycles=[];
      const H=this.halfedges.length;
      for (let h0=0; h0<H; h0++){
        if (visited.has(h0)) continue;
        const seq_h=[], seq_nodes=[];
        let h=h0;
        while(true){
          if (visited.has(h)) break;
          visited.add(h);
          seq_h.push(h);
          const he=this.halfedges[h];
          seq_nodes.push(he.head);
          h=this.successor(h, use_pending);
          if (h===h0){ seq_nodes.push(this.halfedges[h].tail); break; }
        }
        const nodes_no_last = seq_nodes.slice(0,-1);
        const simple = (new Set(nodes_no_last)).size === nodes_no_last.length;
        const poly = nodes_no_last.map(nid=>this.nodes[nid].pos);
        cycles.push({halfedges:seq_h, nodes:seq_nodes, simple, poly, area:shoelace_area(poly)});
      }
      return cycles;
    }
    static _canon_cycle_nodes(nodes){
      if (!nodes || !nodes.length) return [];
      const arr=(nodes[0]===nodes[nodes.length-1]) ? nodes.slice(0,-1) : nodes.slice();
      const n=arr.length, candidates=[];
      for (const rev of [false,true]){
        const seq=rev? arr.slice().reverse():arr.slice();
        let minIdx=0; for (let i=1;i<n;i++) if (seq[i]<seq[minIdx]) minIdx=i;
        const rot=seq.slice(minIdx).concat(seq.slice(0,minIdx));
        candidates.push(rot.join(","));
      }
      return candidates.sort()[0].split(",").map(s=>+s);
    }
    is_jordan_cycle(c, use_pending=false){
      for (const nid of new Set(c.nodes.slice(0,-1))){
        const node=this.nodes[nid];
        if (node.type==='cross'){
          let state=this.cross_state[nid];
          if (use_pending && this.pending_cross && this.pending_cross[0]===nid) state=this.pending_cross[1];
          if (state==='X') return false;
        }
      }
      const r_map=this._cross_radii();
      const poly_sm=this._smoothed_cycle_poly(c.nodes.slice(0,-1), r_map);
      if (poly_sm.length<3) return false;
      if (Math.abs(shoelace_area(poly_sm))<=1e-9) return false;
      c.poly_sm=poly_sm; return true;
    }
    _update_seen_cycles(){
      const cycles=this.enumerate_cycles(false);
      this.seen_simple_cycles.clear();
      for (const c of cycles){
        if (this.is_jordan_cycle(c)){
          const key=SpliceGame._canon_cycle_nodes(c.nodes.slice(0,-1)).join(",");
          this.seen_simple_cycles.add(key);
        }
      }
    }
    commit_and_score(){
      if (!this.pending_cross) return {moved:false, awards:[]};
      const [nid,new_state]=this.pending_cross;
      this.cross_state[nid]=new_state;
      this.pending_cross=null;

      const r_map=this._cross_radii();
      const cycles_after=this.enumerate_cycles(false);

      const new_jordan=[];
      for (const c of cycles_after){
        if (this.is_jordan_cycle(c)){
          const key=SpliceGame._canon_cycle_nodes(c.nodes.slice(0,-1)).join(",");
          if (!this.seen_simple_cycles.has(key)){
            c.poly_sm=this._smoothed_cycle_poly(c.nodes.slice(0,-1), r_map);
            new_jordan.push(c);
          }
        }
      }
      const contains_claimed = poly_sm => this.claimed.some(cl => point_in_poly(cl.label, poly_sm));
      const new_awards=[];

      for (const c of new_jordan){
        if (!c.poly_sm) c.poly_sm=this._smoothed_cycle_poly(c.nodes.slice(0,-1), r_map);
      }

      const new_interior_pts = new_jordan.map(c=>interior_point(c.poly_sm));
      const new_areas = new_jordan.map(c=>Math.abs(shoelace_area(c.poly_sm)));
      const contains_other_new = new Array(new_jordan.length).fill(false);
      for (let i=0;i<new_jordan.length;i++){
        const poly_i = new_jordan[i].poly_sm;
        for (let j=0;j<new_jordan.length;j++){
          if (i===j) continue;
          const pt = new_interior_pts[j];
          if (!pt) continue;
          if (new_areas[i] > new_areas[j] && point_in_poly(pt, poly_i)){
            contains_other_new[i]=true;
            break;
          }
        }
      }

      new_jordan.forEach((c, idx)=>{
        const poly_sm=c.poly_sm || this._smoothed_cycle_poly(c.nodes.slice(0,-1), r_map);
        if (poly_sm.length>=3 && !contains_claimed(poly_sm) && !contains_other_new[idx]){
          const desired_margin_px=12.0;
          const margin_world = (this.port_radius_world!=null)
            ? (desired_margin_px/GLOBAL_PORT_RADIUS_PX)*this.port_radius_world
            : 0.5;
          const lbl=this._best_label_point(poly_sm, margin_world);
          this.claimed.push({nodes:c.nodes.slice(), poly:poly_sm.slice(), label:lbl.slice(), owner:this.player, outside:false});
          this.scores[this.player]+=1;
          new_awards.push({type:'disk', area:shoelace_area(poly_sm), player:this.player});
        }
      });

      const cross_ids = []; for (let i=0;i<this.nodes.length;i++) if (this.nodes[i].type==='cross') cross_ids.push(i);
      if (cross_ids.every(cid=>this.cross_state[cid]!=='X')){
        const jordan_after = cycles_after.filter(c=>this.is_jordan_cycle(c));
        if (jordan_after.length===1){
          const onlyc=jordan_after[0];
          const already_exact = this.claimed.some(cl=>{
            const a=SpliceGame._canon_cycle_nodes(cl.nodes.slice(0,-1)).join(",");
            const b=SpliceGame._canon_cycle_nodes(onlyc.nodes.slice(0,-1)).join(",");
            return a===b;
          });
          if (!already_exact){
            const poly_sm=this._smoothed_cycle_poly(onlyc.nodes.slice(0,-1), r_map);
            if (poly_sm.length>=3){
              const desired_margin_px=12.0;
              const margin_world = (this.port_radius_world!=null)
                ? (desired_margin_px/GLOBAL_PORT_RADIUS_PX)*this.port_radius_world
                : 0.5;
              const lbl=this._best_label_point(poly_sm, margin_world);
              this.claimed.push({nodes:onlyc.nodes.slice(), poly:poly_sm.slice(), label:lbl.slice(), owner:this.player, outside:true});
              this.scores[this.player]+=1;
              new_awards.push({type:'outside', area:shoelace_area(poly_sm), player:this.player});
            }
          }
        }
      }
      this._update_seen_cycles();
      this.player ^= 1;
      return {moved:true, awards:new_awards};
    }

    find_nearest_crossing(sx, sy, world_to_screen, max_pix=14){
      let best=null, best_d=Infinity;
      for (let nid=0;nid<this.nodes.length;nid++){
        const n=this.nodes[nid];
        if (n.type!=='cross') continue;
        if (this.cross_state[nid]!=='X') continue;
        const [px,py]=world_to_screen(n.pos);
        const d=Math.hypot(px-sx, py-sy);
        if (d<best_d && d<=max_pix){ best_d=d; best=nid; }
      }
      return best;
    }

    cycle_crossing_state(nid){
      let cur=this.cross_state[nid];
      if (this.pending_cross && this.pending_cross[0]===nid) cur=this.pending_cross[1];
      const nxt={X:'A',A:'B',B:'X'}[cur];
      if (nxt!==this.cross_state[nid]) this.pending_cross=[nid,nxt];
      else this.pending_cross=null;
    }

    _cross_radii(){
      if (this.port_radius_world!=null){
        const R=+this.port_radius_world, out={};
        for (let nid=0;nid<this.nodes.length;nid++) if (this.nodes[nid].type==='cross') out[nid]=R;
        return out;
      }
      // fallback heuristic (rarely used; renderer sets it on boot)
      const r_map={};
      for (let nid=0;nid<this.nodes.length;nid++){
        const n=this.nodes[nid]; if (n.type!=='cross') continue;
        const [cx,cy]=n.pos;
        const lens=[];
        for (const [eid,_ang] of n.angles){
          const e=this.edges[eid];
          const other=(e.u===nid)? e.p_v : e.p_u;
          lens.push(Math.hypot(other[0]-cx, other[1]-cy));
        }
        lens.sort((a,b)=>a-b);
        const L_ref=lens.length>=2? lens[1] : lens[0];
        const dirs=n.angles.map(a=>((a[1]+Math.PI)%Math.PI)).sort((a,b)=>a-b);
        let delta=Math.abs(dirs[1]-dirs[0]); delta=Math.min(delta, Math.PI-delta);
        const angle_factor=0.80+0.20*Math.min(1.0, delta/(Math.PI/2));
        r_map[nid]=Math.min(0.95*L_ref, 0.55*L_ref)*angle_factor;
      }
      return r_map;
    }

    _smoothed_cycle_poly(nodes_cycle, r_map, alpha=0.55, steps=48){
      if (!nodes_cycle || !nodes_cycle.length) return [];
      const n=nodes_cycle.length;
      const out=[];
      const append_pt=p=>{
        const m=out.length;
        if (m===0){ out.push([+p[0],+p[1]]); return; }
        const [x0,y0]=out[m-1];
        if (Math.abs(p[0]-x0)>1e-10 || Math.abs(p[1]-y0)>1e-10) out.push([+p[0],+p[1]]);
      };

      for (let k=0;k<n;k++){
        const u=nodes_cycle[k], v=nodes_cycle[(k+1)%n];
        const node_u=this.nodes[u], node_v=this.nodes[v];

        let Pstart;
        if (node_u.type==='cross' && this.cross_state[u]!=='X'){
          const Ru=r_map[u]||0.0; Pstart=this._port_on_edge_world(u,v,Ru)[0];
        }else Pstart=this.nodes[u].pos;

        let Pend, t_in=null;
        if (node_v.type==='cross' && this.cross_state[v]!=='X'){
          const Rv=r_map[v]||0.0; const r=this._port_on_edge_world(v,u,Rv); Pend=r[0]; t_in=r[1];
        }else{ Pend=this.nodes[v].pos; t_in=null; }

        if (k===0) append_pt(Pstart);
        append_pt(Pend);

        if (node_v.type==='cross' && this.cross_state[v]!=='X'){
          const w=nodes_cycle[(k+2)%n];
          const Rv=r_map[v]||0.0;
          const r2=this._port_on_edge_world(v,w,Rv); const Pout=r2[0], t_out=r2[1];

          const P0=[+Pend[0], +Pend[1]], P3=[+Pout[0], +Pout[1]];
          const Tin=t_in? [t_in[0],t_in[1]] : [0,0], Tout=[t_out[0],t_out[1]];
          const P1=[P0[0]-alpha*Rv*Tin[0], P0[1]-alpha*Rv*Tin[1]];
          const P2=[P3[0]-alpha*Rv*Tout[0], P3[1]-alpha*Rv*Tout[1]];
          for (let tt=1; tt<steps-1; tt++){
            const t=tt/(steps-1), mt=1-t;
            const Q=[
              (mt**3)*P0[0] + 3*(mt**2)*t*P1[0] + 3*mt*(t**2)*P2[0] + (t**3)*P3[0],
              (mt**3)*P0[1] + 3*(mt**2)*t*P1[1] + 3*mt*(t**2)*P2[1] + (t**3)*P3[1],
            ];
            append_pt(Q);
          }
        }
      }
      if (out.length>2){
        const simp=[out[0]];
        for (let i=1;i<out.length-1;i++){
          const a=simp[simp.length-1], b=out[i], c=out[i+1];
          if (Math.hypot(b[0]-a[0], b[1]-a[1])<1e-9) continue;
          const area2=Math.abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]));
          if (area2<1e-12) continue;
          simp.push(b);
        }
        simp.push(out[out.length-1]);
        return simp;
      }
      return out;
    }
  }

  // ------------------------------- Renderer -----------------------------------

  class Renderer {
    constructor(game, canvas){
      this.game=game;
      this.canvas=canvas;
      this.ctx=canvas.getContext("2d");

      this._compute_view_transform();

      this.port_radius_world = GLOBAL_PORT_RADIUS_PX/this.scale;
      this.game.port_radius_world = this.port_radius_world;
      this.game._update_seen_cycles();

      this.pendingToast=null; this.toastTimer=0;
    }

    _compute_view_transform(){
      const pts=this.game.nodes.map(n=>n.pos);
      let minx=Infinity,miny=Infinity,maxx=-Infinity,maxy=-Infinity;
      for (const [x,y] of pts){
        if (x<minx) minx=x; if (x>maxx) maxx=x;
        if (y<miny) miny=y; if (y>maxy) maxy=y;
      }
      const dx=(maxx-minx)||1.0, dy=(maxy-miny)||1.0;
      minx -= 0.08*dx; maxx += 0.08*dx;
      miny -= 0.08*dy; maxy += 0.08*dy;

      const cssW = this.canvas.clientWidth || this.canvas.width;
      const cssH = this.canvas.clientHeight || this.canvas.height;
      // Use the CSS pixel size for layout; the context is already DPR-scaled
      const avail_w=cssW, avail_h=cssH;
      const sx=avail_w/(maxx-minx), sy=avail_h/(maxy-miny);
      this.scale=Math.min(sx,sy);

      const world_w=(maxx-minx), world_h=(maxy-miny);
      const draw_w=this.scale*world_w, draw_h=this.scale*world_h;
      const offx=(avail_w-draw_w)*0.5, offy=(avail_h-draw_h)*0.5;

      this.minx=minx; this.maxy=maxy;
      this.offx=offx; this.offy=offy;

      this.cross_r_world = 0.045*Math.max(world_w, world_h);
    }

    world_to_screen(p){
      const sx = this.offx + (p[0]-this.minx)*this.scale;
      const sy = this.offy + (this.maxy - p[1])*this.scale;
      return [Math.round(sx), Math.round(sy)];
    }
    world_to_screen_f(p){
      const sx = this.offx + (p[0]-this.minx)*this.scale;
      const sy = this.offy + (this.maxy - p[1])*this.scale;
      return [sx, sy];
    }
    screen_to_world(p){
      const x = (p[0]-this.offx)/this.scale + this.minx;
      const y = this.maxy - (p[1]-this.offy)/this.scale;
      return [x,y];
    }

    _next_edge_along_strand(edge, at_node){
      const idx_in=this.game.edge_endpoint_stub[`${edge.id},${at_node}`];
      const node=this.game.nodes[at_node];
      const out_idx = (node.type==='vertex') ? (1-idx_in) : ((idx_in+2)%4);
      const eid2=node.angles[out_idx][0];
      const e2=this.game.edges[eid2];
      let p0,p1,nxt;
      if (e2.u===at_node){ p0=this.game.nodes[at_node].pos; p1=e2.p_v; nxt=e2.v; }
      else{ p0=this.game.nodes[at_node].pos; p1=e2.p_u; nxt=e2.u; }
      return [e2, [p0[0],p0[1]], [p1[0],p1[1]], nxt];
    }

    _circle_exit_on_stub(nid, eid, R_px){
      const Cw=this.game.nodes[nid].pos.slice();
      const Rw=R_px/this.scale;
      let e=this.game.edges[eid], p0,p1,cur_node;
      if (e.u===nid){ p0=e.p_u.slice(); p1=e.p_v.slice(); cur_node=e.v; }
      else{ p0=e.p_v.slice(); p1=e.p_u.slice(); cur_node=e.u; }
      let steps=0;
      while (steps<500){
        const a=vsub(p1,p0);
        const A=dot(a[0],a[1],a[0],a[1]);
        const b=vsub(p0,Cw);
        const B=2*dot(a[0],a[1],b[0],b[1]);
        const C=dot(b[0],b[1],b[0],b[1])-Rw*Rw;
        const disc=B*B-4*A*C;
        if (disc>=0){
          const sq=Math.sqrt(disc);
          const t1=(-B - sq)/(2*A), t2=(-B + sq)/(2*A);
          const cands=[t1,t2].filter(t=>0<=t && t<=1);
          if (cands.length){
            const pos=cands.filter(t=>t>=1e-9);
            const t=pos.length? Math.min(...pos) : Math.min(...cands);
            const Pw=vadd(p0, vmul(a,t));
            const u=vnorm(a);
            const upx=[u[0], -u[1]];
            const Ppx=this.world_to_screen_f(Pw);
            return [Ppx, upx];
          }
        }
        [e,p0,p1,cur_node]=this._next_edge_along_strand(e, cur_node);
        steps++;
      }
      let u=vnorm(vsub(p1,p0));
      const Pw=vadd(Cw, vmul(u,Rw));
      return [this.world_to_screen_f(Pw), [u[0],-u[1]]];
    }

    showToast(msg, ms=1500){
      this.pendingToast=msg; this.toastTimer=ms;
      const el=document.getElementById("toast"); el.textContent=msg; el.hidden=false;
    }

    draw(dt){
      const ctx=this.ctx;
      const w=this.canvas.clientWidth || this.canvas.width;
      const h=this.canvas.clientHeight || this.canvas.height;

      // play background (covers whole canvas)
      ctx.fillStyle=PLAY_BG;
      ctx.fillRect(0,0,w,h);

      // compute crossing list
      const cross_ids=[]; for (let i=0;i<this.game.nodes.length;i++) if (this.game.nodes[i].type==='cross') cross_ids.push(i);

      // local radii (used for halos)
      const px_floor=30, px_cap=48;
      const r_floor=px_floor/this.scale, r_cap=px_cap/this.scale;
      const local_r={}, r_eff={};
      for (const nid of cross_ids){
        const n=this.game.nodes[nid], [cx,cy]=n.pos;
        const lens=[];
        for (const [eid,_ang] of n.angles){
          const e=this.game.edges[eid];
          const other=(e.u===nid)? e.p_v : e.p_u;
          lens.push(Math.hypot(other[0]-cx, other[1]-cy));
        }
        lens.sort((a,b)=>a-b);
        const L_ref=(lens[1]??lens[0]);
        const dirs=n.angles.map(a=>((a[1]+Math.PI)%Math.PI)).sort((a,b)=>a-b);
        let delta=Math.abs(dirs[1]-dirs[0]); delta=Math.min(delta, Math.PI-delta);
        const angle_factor = 0.80+0.20*Math.min(1.0, delta/(Math.PI/2));
        let r_try=Math.max(r_floor, this.cross_r_world, 0.55*L_ref);
        r_try=Math.min(r_try, r_cap);
        let r_fin=Math.max(Math.min(0.95*L_ref, r_try), r_floor);
        local_r[nid]=r_fin*angle_factor;
      }
      const pad_world=(Math.max(EDGE_W,SMOOTH_W)*0.5 + FILL_SAFETY_PX)/this.scale;
      for (const k of Object.keys(local_r)) r_eff[k]=local_r[k]+pad_world;

      // 1) black polyline
      ctx.strokeStyle=EDGE_COLOR; ctx.lineWidth=EDGE_W; ctx.lineCap="round";
      for (const e of this.game.edges){
        const A=this.world_to_screen(e.p_u), B=this.world_to_screen(e.p_v);
        if (A[0]!==B[0] || A[1]!==B[1]){
          ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.lineTo(B[0],B[1]); ctx.stroke();
        }
      }

      // 2) trim around crossings + compute ports
      const ports=new Map();
      for (const nid of cross_ids){
        const [cx,cy]=this.game.nodes[nid].pos;
        const [cpx,cpy]=this.world_to_screen([cx,cy]);
        const OVERPAINT_PX = Math.floor(Math.max(EDGE_W,SMOOTH_W)/2)+2;
        const erase_r = Math.max(1, Math.round(GLOBAL_PORT_RADIUS_PX - OVERPAINT_PX));
        ctx.fillStyle=PLAY_BG; ctx.beginPath(); ctx.arc(cpx,cpy,erase_r,0,Math.PI*2); ctx.fill();

        const inc_sorted=this.game.nodes[nid].angles;
        for (let s=0;s<inc_sorted.length;s++){
          const [eid,_ang]=inc_sorted[s];
          const [Ppx, upx]=this._circle_exit_on_stub(nid, eid, GLOBAL_PORT_RADIUS_PX);
          ports.set(`${nid},${s}`, [Ppx, upx]);
        }
      }

      // 3) red smoothing
      const draw_cubic_dir = (A,B,tA,tB,scale_px,alpha=0.55) =>{
        ctx.strokeStyle=SMOOTH_COLOR; ctx.lineWidth=SMOOTH_W; ctx.lineCap="round";
        const P1=[ A[0] + alpha*scale_px*tA[0], A[1] + alpha*scale_px*tA[1] ];
        const P2=[ B[0] - alpha*scale_px*tB[0], B[1] - alpha*scale_px*tB[1] ];
        ctx.beginPath(); ctx.moveTo(A[0],A[1]); ctx.bezierCurveTo(P1[0],P1[1], P2[0],P2[1], B[0],B[1]); ctx.stroke();
      };
      const draw_bezier_px=(P0,P3,u_in,u_out,R_px,alpha=0.55)=>{
        const P1=[ P0[0] - alpha*R_px*u_in[0], P0[1] - alpha*R_px*u_in[1] ];
        const P2=[ P3[0] - alpha*R_px*u_out[0], P3[1] - alpha*R_px*u_out[1] ];
        ctx.strokeStyle=SMOOTH_COLOR; ctx.lineWidth=SMOOTH_W; ctx.lineCap="round";
        ctx.beginPath(); ctx.moveTo(P0[0],P0[1]); ctx.bezierCurveTo(P1[0],P1[1], P2[0],P2[1], P3[0],P3[1]); ctx.stroke();
      };

      for (const nid of cross_ids){
        let state=this.game.cross_state[nid];
        if (this.game.pending_cross && this.game.pending_cross[0]===nid) state=this.game.pending_cross[1];
        const ends=[0,1,2,3].map(s=>ports.get(`${nid},${s}`));
        if (state==='X'){
          const Cpx=this.world_to_screen_f(this.game.nodes[nid].pos);
          for (const [i,j] of [[0,2],[1,3]]){
            const [P0,u0]=ends[i], [P3,u3]=ends[j];
            draw_cubic_dir(P0, Cpx, [-u0[0],-u0[1]], [u0[0],u0[1]], GLOBAL_PORT_RADIUS_PX);
            draw_cubic_dir(Cpx, P3, [u3[0],u3[1]], [-u3[0],-u3[1]], GLOBAL_PORT_RADIUS_PX);
          }
        }else if (state==='A'){
          for (const [i,j] of [[0,1],[2,3]]){
            const [P0,u_in]=ends[i], [P3,u_out]=ends[j];
            draw_bezier_px(P0,P3,u_in,u_out,GLOBAL_PORT_RADIUS_PX);
          }
        }else if (state==='B'){
          for (const [i,j] of [[1,2],[3,0]]){
            const [P0,u_in]=ends[i], [P3,u_out]=ends[j];
            draw_bezier_px(P0,P3,u_in,u_out,GLOBAL_PORT_RADIUS_PX);
          }
        }
      }

      // 4) claimed fills
      for (const cl of this.game.claimed){
        const poly=cl.poly;
        if (poly && poly.length>=3){
          const pts=poly.map(p=>this.world_to_screen(p));
          ctx.fillStyle=(cl.owner===0)? CLAIM_FILL_COLOR_P1 : CLAIM_FILL_COLOR_P2;
          ctx.beginPath(); ctx.moveTo(pts[0][0],pts[0][1]);
          for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]);
          ctx.closePath(); ctx.fill();
        }
      }

      // 5) crossings (dots + pending halo)
      for (const nid of cross_ids){
        const [px,py]=this.world_to_screen(this.game.nodes[nid].pos);
        const pending = (this.game.pending_cross && this.game.pending_cross[0]===nid);
        let eff_state=this.game.cross_state[nid]; if (pending) eff_state=this.game.pending_cross[1];

        if (pending && (eff_state==='A' || eff_state==='B')){
          const base_r_px = (r_eff[nid]??this.cross_r_world) * this.scale;
          const halo_r_px = Math.round(Math.max(22, Math.min(90, base_r_px*1.25)));
          ctx.strokeStyle=PENDING_HALO; ctx.lineWidth=3;
          ctx.beginPath(); ctx.arc(px,py,halo_r_px,0,Math.PI*2); ctx.stroke();
          continue;
        }
        if (eff_state==='X'){
          if (pending){
            ctx.strokeStyle=PENDING_HALO; ctx.lineWidth=2;
            ctx.beginPath(); ctx.arc(px,py,9,0,Math.PI*2); ctx.stroke();
          }
          ctx.fillStyle=CROSS_DOT_COLOR;
          ctx.beginPath(); ctx.arc(px,py,4,0,Math.PI*2); ctx.fill();
        }
      }

      // score label (HTML)
      const scoreP1 = document.getElementById("score-p1");
      const scoreP2 = document.getElementById("score-p2");
      const turnBubble = document.getElementById("score-turn");
      const turnValue = turnBubble ? turnBubble.querySelector(".score-value") : null;
      const p1Bubble = document.querySelector("#score .player1");
      const p2Bubble = document.querySelector("#score .player2");

      if (scoreP1) scoreP1.textContent = this.game.scores[0];
      if (scoreP2) scoreP2.textContent = this.game.scores[1];
      if (turnValue) turnValue.textContent = `Player ${this.game.player+1}`;

      if (p1Bubble && p2Bubble){
        p1Bubble.classList.toggle("active", this.game.player===0);
        p2Bubble.classList.toggle("active", this.game.player===1);
      }
      if (turnBubble){
        turnBubble.classList.toggle("player1-turn", this.game.player===0);
        turnBubble.classList.toggle("player2-turn", this.game.player===1);
      }

      if (this.toastTimer>0){
        this.toastTimer -= dt;
        if (this.toastTimer<=0){ document.getElementById("toast").hidden=true; this.pendingToast=null; }
      }
    }
  }

  // -------------------------------- App ---------------------------------------

  class App {
    constructor(seed=null, min_cross_sep=1.5, min_cross_angle_deg=25.0){
      this.canvas = document.getElementById("canvas");
      this._resizeCanvas(); // sets canvas size responsively

      // Robust creation with backoff (ensures you see a curve even on unlucky seeds)
      this.game = this._makeGameWithBackoff(seed, min_cross_sep, min_cross_angle_deg);
      this.renderer = new Renderer(this.game, this.canvas);
      this.undo_stack=[];
      this._bindEvents();
      window.addEventListener("resize", ()=>{ this._resizeCanvas(); this.renderer._compute_view_transform(); });
    }

    _makeGameWithBackoff(seed, sep, angDeg){
      const attempts = [
        {sep, ang: angDeg},
        {sep: Math.max(1.2, sep*0.8), ang: Math.max(20, angDeg-5)},
        {sep: Math.max(1.0, sep*0.66), ang: Math.max(15, angDeg-10)},
      ];
      for (const a of attempts){
        try{ return new SpliceGame(seed, a.sep, a.ang); }catch(e){}
      }
      // final fallback: new seed + relaxed
      return new SpliceGame(null, 1.0, 15.0);
    }

    _resizeCanvas(){
      const container = document.getElementById("play-area");
      const hud = document.getElementById("hud");
      const title = document.getElementById("title");
      const dpr = window.devicePixelRatio || 1;

      const maxSide = 900; // logical cap
      const cw = Math.min(container.clientWidth, maxSide);
      // Fit height within viewport so no scrolling needed
      const availH = Math.max(400, window.innerHeight - hud.offsetHeight - title.offsetHeight - 40);
      const sideCSS = Math.max(360, Math.min(cw, availH)); // square

      // set CSS size
      this.canvas.style.width = sideCSS + "px";
      this.canvas.style.height = sideCSS + "px";

      // set internal pixel size (DPR aware)
      this.canvas.width  = Math.floor(sideCSS * dpr);
      this.canvas.height = Math.floor(sideCSS * dpr);

      // reset transform to avoid blurry lines after DPR changes
      const ctx=this.canvas.getContext("2d");
      ctx.setTransform(1,0,0,1,0,0);
      ctx.scale(dpr, dpr); // draw in CSS pixels
    }

    snapshot(){
      const cs={...this.game.cross_state};
      const player=this.game.player;
      const scores=[...this.game.scores];
      const claimed=this.game.claimed.map(c=>({
        nodes:[...c.nodes],
        poly:c.poly.map(p=>[+p[0],+p[1]]),
        label:[+c.label[0], +c.label[1]],
        owner:c.owner, outside:c.outside
      }));
      const seen=new Set(this.game.seen_simple_cycles);
      return {cs, player, scores, claimed, seen};
    }
    restore(snap){
      const {cs, player, scores, claimed, seen} = snap;
      this.game.cross_state={...cs};
      this.game.player=player;
      this.game.scores=[...scores];
      this.game.claimed = claimed.map(c=>({
        nodes:[...c.nodes],
        poly:c.poly.map(p=>[+p[0],+p[1]]),
        label:[+c.label[0], +c.label[1]],
        owner:c.owner, outside:c.outside
      }));
      this.game.seen_simple_cycles=new Set(seen);
      this.game.pending_cross=null;
    }

    _bindEvents(){
      document.getElementById("btn-confirm").addEventListener("click", ()=>{
        if (this.game.pending_cross){
          this.undo_stack.push(this.snapshot());
          const {awards=[]}=this.game.commit_and_score();
          if (awards.length){
            const msg = awards.map(a => `+1 ${a.type==='outside'?'outside':'disk'} (P${a.player+1})`).join("; ");
            this.renderer.showToast(msg, 1500);
          }
        }
      });
      document.getElementById("btn-new").addEventListener("click", ()=>{
        this.game = this._makeGameWithBackoff(null, this.game.min_cross_sep, (this.game.min_cross_angle_rad*180/Math.PI));
        this.renderer = new Renderer(this.game, this.canvas);
        this.undo_stack=[];
      });
      document.getElementById("btn-undo").addEventListener("click", ()=>{
        if (this.undo_stack.length){
          this.restore(this.undo_stack.pop());
        }
      });

      this.canvas.addEventListener("mousedown", (ev)=>{
        if (ev.button!==0) return;
        const rect=this.canvas.getBoundingClientRect();
        // because we scaled the context by DPR, use CSS pixels here
        const sx = ev.clientX - rect.left;
        const sy = ev.clientY - rect.top;
        const nid=this.game.find_nearest_crossing(sx, sy, p=>this.renderer.world_to_screen(p), 28);
        if (nid==null) return;
        if (this.game.cross_state[nid]!=='X') return;
        if (this.game.pending_cross){
          const [nid_pending, pstate]=this.game.pending_cross;
          if (nid!==nid_pending && pstate!==this.game.cross_state[nid_pending]){
            // ignore
          }else{
            this.game.cycle_crossing_state(nid);
          }
        }else{
          this.game.cycle_crossing_state(nid);
        }
      });

      window.addEventListener("keydown", (ev)=>{
        if (ev.key==="Enter" || ev.key==="c"){
          if (this.game.pending_cross){
            this.undo_stack.push(this.snapshot());
            const {awards=[]}=this.game.commit_and_score();
            if (awards.length){
              const msg = awards.map(a => `+1 ${a.type==='outside'?'outside':'disk'} (P${a.player+1})`).join("; ");
              this.renderer.showToast(msg, 1500);
            }
          }
        }else if (ev.key==="n"){
          this.game = this._makeGameWithBackoff(null, this.game.min_cross_sep, (this.game.min_cross_angle_rad*180/Math.PI));
          this.renderer = new Renderer(this.game, this.canvas);
          this.undo_stack=[];
        }else if (ev.key==="u" || ev.key==="Backspace"){
          if (this.undo_stack.length) this.restore(this.undo_stack.pop());
        }
      });
    }

    run(){
      let last=performance.now();
      const loop=(t)=>{
        const dt=t-last; last=t;
        this.renderer.draw(dt);
        requestAnimationFrame(loop);
      };
      requestAnimationFrame(loop);
    }
  }

  // ------------------------------ Boot ----------------------------------------
  // Build with defaults similar to your Pygame main()
  const app = new App(null, 1.5, 25.0);
  app.run();
})();
