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
  const DEFAULT_CROSS_COUNT = 5;
  const RAYMOND_EGG_ENABLED = false; // disabled per request

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
    constructor(seed=null, min_cross_sep=0.6, min_cross_angle_deg=25.0, target_crosses=10){
      this.min_cross_sep = +min_cross_sep;
      this.min_cross_angle_rad = (Math.PI/180)*(+min_cross_angle_deg);
      this.target_crosses = Math.max(4, Math.min(60, Math.round(+target_crosses || 10)));
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
      this.finished=false;
      this._update_seen_cycles();
      this.port_radius_world = null; // renderer will set
    }

    _make_good_random_graph(){
      let best=null;
      let bestDiff=Infinity;
      const complexity=Math.max(0, this.target_crosses-6);
      const extraModes=Math.min(4, Math.floor(complexity/3));
      const modes=4+extraModes;
      const baseSamples=140 + Math.min(120, complexity*12);
      const baseRadius=Math.max(5.0, 2.0*this.min_cross_sep);
      const radiusScale=1 + 0.12*Math.max(0, this.target_crosses-8);
      const maxTries = this.target_crosses>=16 ? 1024 : this.target_crosses>=10 ? 800 : 512;
      for (let tries=0; tries<maxTries; tries++){
        const jitter=0.85 + RAND()*0.6;
        const samples = baseSamples + Math.floor(RAND()*24);
        const r = baseRadius * radiusScale * jitter;
        const poly = random_fourier_closed_polyline(samples, modes, r);
        const res = build_immersed_graph(poly, this.min_cross_sep, this.min_cross_angle_rad);
        if (res){
          const [nodes, edges] = res;
          const n_cross = nodes.reduce((a,n)=>a+(n.type==='cross'?1:0),0);
          if (n_cross>=4){
            const diff=Math.abs(n_cross - this.target_crosses);
            if (diff===0) return [nodes, edges];
            if (!best || diff<bestDiff || (diff===bestDiff && n_cross>best.crosses)){
              best={nodes, edges, crosses:n_cross};
              bestDiff=diff;
            }
          }
        }
      }
      if (best) return [best.nodes, best.edges];
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
        const jordan_after=[];
        for (const cycle of cycles_after){
          if (this.is_jordan_cycle(cycle)){ jordan_after.push(cycle); }
        }
        const canonOf = nodes => SpliceGame._canon_cycle_nodes(nodes.slice(0,-1)).join(",");
        const existingOutside=this.claimed.some(cl=>cl.outside);
        if (!existingOutside){
          const claimedOutsideKeys=new Set(
            this.claimed.filter(cl=>cl.outside).map(cl=>canonOf(cl.nodes))
          );
          let bestOuter=null;
          for (const c of jordan_after){
            const key=canonOf(c.nodes);
            if (claimedOutsideKeys.has(key)) continue;
            const poly_sm=c.poly_sm || this._smoothed_cycle_poly(c.nodes.slice(0,-1), r_map);
            if (!poly_sm || poly_sm.length<3) continue;
            const enclosesAllClaims = this.claimed.every(cl => cl.outside || point_in_poly(cl.label, poly_sm));
            if (!enclosesAllClaims) continue;
            const area=Math.abs(shoelace_area(poly_sm));
            if (!bestOuter || area>bestOuter.area){
              bestOuter={cycle:c, key, poly:poly_sm, area};
            }
          }
          if (bestOuter){
            const desired_margin_px=12.0;
            const margin_world = (this.port_radius_world!=null)
              ? (desired_margin_px/GLOBAL_PORT_RADIUS_PX)*this.port_radius_world
              : 0.5;
            const lbl=this._best_label_point(bestOuter.poly, margin_world);
            this.claimed.push({nodes:bestOuter.cycle.nodes.slice(), poly:bestOuter.poly.slice(), label:lbl.slice(), owner:this.player, outside:true});
            this.scores[this.player]+=1;
            new_awards.push({type:'outside', area:shoelace_area(bestOuter.poly), player:this.player});
          }
        }
      }
      this._update_seen_cycles();
      const lastMover=this.player;
      const result={moved:true, awards:new_awards};

      this.finished=this.isGameOver();
      if (this.finished){
        const winner=this.currentWinner();
        result.gameOver={
          winner,
          scores:[...this.scores],
          tie:winner==null,
          lastMover,
        };
      }

      this.player ^= 1;
      return result;
    }

    cloneState(){
      const clone=Object.create(SpliceGame.prototype);
      clone.min_cross_sep=this.min_cross_sep;
      clone.min_cross_angle_rad=this.min_cross_angle_rad;
      clone.target_crosses=this.target_crosses;
      clone.seed=this.seed;
      clone.nodes=this.nodes;
      clone.edges=this.edges;
      clone.halfedges=this.halfedges;
      clone.stub_to_he=this.stub_to_he;
      clone.edge_to_he=this.edge_to_he;
      clone.edge_endpoint_stub=this.edge_endpoint_stub;
      clone.port_radius_world=this.port_radius_world;
      clone.cross_state={...this.cross_state};
      clone.pending_cross=this.pending_cross ? [this.pending_cross[0], this.pending_cross[1]] : null;
      clone.player=this.player;
      clone.scores=[...this.scores];
      clone.claimed=this.claimed.map(c=>({
        nodes:[...c.nodes],
        poly:c.poly.map(p=>[+p[0], +p[1]]),
        label:[+c.label[0], +c.label[1]],
        owner:c.owner,
        outside:c.outside
      }));
      clone.seen_simple_cycles=new Set(this.seen_simple_cycles);
      clone.finished=this.finished;
      clone.crossCount=this.crossCount;
      return clone;
    }

    isGameOver(){
      for (let nid=0; nid<this.nodes.length; nid++){
        const node=this.nodes[nid];
        if (node.type==='cross' && this.cross_state[nid]==='X') return false;
      }
      return true;
    }

    currentWinner(){
      if (this.scores[0]===this.scores[1]) return null;
      return (this.scores[0]>this.scores[1]) ? 0 : 1;
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
      if (this.finished) return;
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
      this.playerNames=["Player 1", "Player 2"];

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
      this.avail_w=avail_w; this.avail_h=avail_h;

      this.cross_r_world = 0.045*Math.max(world_w, world_h);
    }

    setPlayerNames(names){
      if (!Array.isArray(names) || names.length<2) return;
      this.playerNames=[`${names[0]}`, `${names[1]}`];
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
        if (!poly || poly.length<3) continue;
        const pts=poly.map(p=>this.world_to_screen(p));
        ctx.fillStyle=(cl.owner===0)? CLAIM_FILL_COLOR_P1 : CLAIM_FILL_COLOR_P2;
        if (cl.outside){
          const w=this.avail_w ?? (this.canvas.clientWidth || this.canvas.width);
          const h=this.avail_h ?? (this.canvas.clientHeight || this.canvas.height);
          ctx.beginPath();
          ctx.rect(0,0,w,h);
          ctx.moveTo(pts[0][0],pts[0][1]);
          for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]);
          ctx.closePath();
          ctx.fill('evenodd');
        }else{
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
      const labelP1 = document.querySelector("#score .player1 .score-label");
      const labelP2 = document.querySelector("#score .player2 .score-label");
      const playerNames = this.playerNames || ["Player 1", "Player 2"];

      if (scoreP1) scoreP1.textContent = this.game.scores[0];
      if (scoreP2) scoreP2.textContent = this.game.scores[1];
      if (labelP1) labelP1.textContent = playerNames[0] ?? "Player 1";
      if (labelP2) labelP2.textContent = playerNames[1] ?? "Player 2";
      const finished=this.game.finished;
      const winner=this.game.currentWinner();
      if (turnValue){
        if (finished){
          if (winner==null){
            turnValue.textContent = "Tie Game";
          }else{
            const winName = playerNames[winner] ?? `Player ${winner+1}`;
            turnValue.textContent = `Victory - ${winName}`;
          }
        }else{
          const curName = playerNames[this.game.player] ?? `Player ${this.game.player+1}`;
          turnValue.textContent = curName;
        }
      }

      if (p1Bubble && p2Bubble){
        p1Bubble.classList.toggle("active", !finished && this.game.player===0);
        p2Bubble.classList.toggle("active", !finished && this.game.player===1);
      }
      if (turnBubble){
        turnBubble.classList.toggle("player1-turn", !finished && this.game.player===0);
        turnBubble.classList.toggle("player2-turn", !finished && this.game.player===1);
        turnBubble.classList.toggle("finished", finished);
        turnBubble.classList.toggle("player1-win", finished && winner===0);
        turnBubble.classList.toggle("player2-win", finished && winner===1);
        turnBubble.classList.toggle("tie", finished && winner==null);
      }

      if (this.toastTimer>0){
        this.toastTimer -= dt;
        if (this.toastTimer<=0){ document.getElementById("toast").hidden=true; this.pendingToast=null; }
      }
    }
  }

  // -------------------------------- App ---------------------------------------

  class App {
    constructor(seed=null, min_cross_sep=2.0, min_cross_angle_deg=25.0){
      this.canvas = document.getElementById("canvas");
      this.playArea = document.getElementById("play-area");
      this._resizeCanvas(); // sets canvas size responsively

      this.baseMinCrossSep = +min_cross_sep;
      this.baseMinCrossAngleDeg = +min_cross_angle_deg;
      this.targetCrossCount = this._readCrossCount();

      // Robust creation with backoff (ensures you see a curve even on unlucky seeds)
      const desired = this.targetCrossCount;
      const initialGame = this._makeGameWithBackoff(seed, this.baseMinCrossSep, this.baseMinCrossAngleDeg);
      const initialCrosses = typeof initialGame.crossCount === "number"
        ? initialGame.crossCount
        : initialGame.nodes.reduce((sum,node)=>sum+(node.type==='cross'?1:0),0);
      const mismatch = initialCrosses !== desired;
      if (mismatch){
        this.targetCrossCount = initialCrosses;
        this._setCrossCountInput(initialCrosses);
      }
      this.game = initialGame;
      this.renderer = new Renderer(this.game, this.canvas);
      this.undo_stack=[];
      this._gameOverHideTimer=null;
      this._gameOverPendingPromise=null;
      this._raymondHoldTimer=null;
      this.mode=null;
      this.aiEnabled=false;
      this.aiPlayerIndex=1;
      this.aiDepth=1;
      this.aiDifficultyNames={1:"Depth 1", 3:"Depth 3", 5:"Depth 5"};
      this._aiThinking=false;
      this._aiPendingTimer=null;
      this.menuScreen=document.getElementById("main-menu");
      this.rulesScreen=document.getElementById("rules-screen");
      this.gameScreen=document.getElementById("game-screen");
      this.menuRootOptions=document.getElementById("menu-root-options");
      this.menuDifficulty=document.getElementById("menu-difficulty");
      this.menuBackBtn=document.getElementById("menu-back");
      this.rulesBackBtn=document.getElementById("rules-back");
      this.menuStartPvpBtn=document.getElementById("menu-start-pvp");
      this.menuStartPvcBtn=document.getElementById("menu-start-pvc");
      this.menuStartOnlineBtn=document.getElementById("menu-start-online");
      this.onlineScreen=document.getElementById("online-setup");
      this.onlineStatusEl=document.getElementById("online-status");
      this.onlineHostCreateBtn=document.getElementById("online-host-create");
      this.onlineHostOffer=document.getElementById("online-host-offer");
      this.onlineHostAnswer=document.getElementById("online-host-answer");
      this.onlineHostApplyBtn=document.getElementById("online-host-apply");
      this.onlineGuestOffer=document.getElementById("online-guest-offer");
      this.onlineGuestConnectBtn=document.getElementById("online-guest-connect");
      this.onlineGuestAnswer=document.getElementById("online-guest-answer");
      this.onlineCancelBtn=document.getElementById("online-cancel");
      this.aiStatusEl=document.getElementById("ai-status");
      this.aiDifficultyLabel=document.getElementById("ai-difficulty-label");
      this.btnConfirm=document.getElementById("btn-confirm");
      this.btnUndo=document.getElementById("btn-undo");
      this.btnNew=document.getElementById("btn-new");
      this.btnReturnMenu=document.getElementById("btn-return-menu");
      this.onlineActive=false;
      this.onlineRole=null;
      this.onlinePlayerIndex=null;
      this.onlinePeer=null;
      this.onlineChannel=null;
      this._suspendOnline=false;
      this._currentSeed=null;
      this._bindEvents();
      window.addEventListener("resize", ()=>{ this._resizeCanvas(); this.renderer._compute_view_transform(); });
      if (mismatch){
        this.renderer.showToast(`Using ${initialCrosses} intersections (closest to requested ${desired}).`, 2400);
      }
      this._showMenuRoot();
    }

    _readCrossCount(){
      const input = document.getElementById("cross-count");
      let value = DEFAULT_CROSS_COUNT;
      if (input){
        const parsed = parseInt(input.value, 10);
        if (!Number.isNaN(parsed)) value = parsed;
        value = Math.max(4, Math.min(30, value));
        if (`${value}` !== input.value) input.value = `${value}`;
      }
      return value;
    }

    _setCrossCountInput(value){
      const input = document.getElementById("cross-count");
      if (input) input.value = `${value}`;
    }

    _makeGameWithBackoff(seed, sep, angDeg){
      const desired = this.targetCrossCount;
      const attempts = [
        {sep, ang: angDeg},
        {sep: Math.max(1.2, sep*0.8), ang: Math.max(20, angDeg-5)},
        {sep: Math.max(1.0, sep*0.66), ang: Math.max(15, angDeg-10)},
      ];
      if (desired>=8){
        attempts.push({sep: Math.max(0.9, sep*0.55), ang: Math.max(12, angDeg-14)});
      }
      if (desired>=12){
        attempts.push({sep: Math.max(0.75, sep*0.45), ang: Math.max(9, angDeg-18)});
      }
      if (desired>=16){
        attempts.push({sep: Math.max(0.65, sep*0.36), ang: Math.max(7, angDeg-24)});
      }
      const MAX_GAME_ATTEMPTS = desired>=16 ? 96 : desired>=10 ? 72 : 48;
      let bestGame=null;
      let bestDiff=Infinity;
      let bestActual=0;
      let lastError=null;
      let nextSeed = seed;
      for (const a of attempts){
        for (let i=0;i<MAX_GAME_ATTEMPTS;i++){
          try{
            const game = new SpliceGame(nextSeed, a.sep, a.ang, desired);
            const actual = game.nodes.reduce((sum,node)=>sum+(node.type==='cross'?1:0),0);
            game.crossCount = actual;
            if (actual === desired) return game;
            const diff = Math.abs(actual - desired);
            if (!bestGame || diff<bestDiff || (diff===bestDiff && actual>bestActual)){
              bestGame = game;
              bestDiff = diff;
              bestActual = actual;
            }
          }catch(e){
            lastError = e;
          }
          nextSeed = null;
        }
      }
      if (bestGame) return bestGame;
      if (lastError) throw lastError;
      // final fallback: new seed + relaxed
      const fallbackGame = new SpliceGame(null, 1.0, 15.0, desired);
      fallbackGame.crossCount = fallbackGame.nodes.reduce((sum,node)=>sum+(node.type==='cross'?1:0),0);
      return fallbackGame;
    }

    _resizeCanvas(){
      const container = document.getElementById("play-area");
      const hud = document.getElementById("hud");
      const header = document.getElementById("game-header");
      const title = document.getElementById("title");
      const dpr = window.devicePixelRatio || 1;

      const maxSide = 900; // logical cap
      const cw = container ? Math.min(container.clientWidth, maxSide) : maxSide;
      const hudHeight = hud ? hud.offsetHeight : 0;
      const headerHeight = header ? header.offsetHeight : (title ? title.offsetHeight : 0);
      const chromeAllowance = 56; // padding + breathing room so controls stay in view
      const availH = Math.max(360, window.innerHeight - hudHeight - headerHeight - chromeAllowance);
      const sideCSS = Math.max(320, Math.min(cw, availH)); // square

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

    _setBoardBlurred(active){
      if (!this.playArea) return;
      this.playArea.classList.toggle("blurred", !!active);
    }

    _setSectionVisible(node, show){
      if (!node) return;
      node.hidden = !show;
    }

    _cancelPendingAI(){
      if (this._aiPendingTimer){
        clearTimeout(this._aiPendingTimer);
        this._aiPendingTimer=null;
      }
      this._aiThinking=false;
      this._setControlsDisabled(false);
    }

    _isInMatch(){
      return this.mode!==null;
    }

    _isComputerTurn(){
      return this.aiEnabled && this.game && this.game.player===this.aiPlayerIndex;
    }

    _isInteractionLocked(){
      if (!this._isInMatch()) return true;
      if (this._aiThinking) return true;
      if (this.onlineActive){
        if (this.onlinePlayerIndex==null) return true;
        if (this.game && this.game.player!==this.onlinePlayerIndex) return true;
      }
      return this._isComputerTurn();
    }

    _setControlsDisabled(disabled){
      if (this.btnConfirm) this.btnConfirm.disabled=!!disabled;
      if (this.btnUndo) this.btnUndo.disabled = this.onlineActive ? true : !!disabled;
    }

    _refreshControlsForTurn(){
      if (this.btnConfirm){
        if (this.onlineActive){
          this.btnConfirm.disabled = this._isInteractionLocked();
        }else if (!this._aiThinking){
          this.btnConfirm.disabled=false;
        }
      }
    }

    _currentPlayerNames(){
      if (this.onlineActive){
        if (this.onlinePlayerIndex===0) return ["You", "Opponent"];
        if (this.onlinePlayerIndex===1) return ["Opponent", "You"];
      }
      if (this.aiEnabled){
        return ["You", "Computer"];
      }
      return ["Player 1", "Player 2"];
    }

    _playerDisplayName(idx){
      const names=this._currentPlayerNames();
      if (idx==null || idx<0 || idx>=names.length) return `Player ${(idx??0)+1}`;
      return names[idx] ?? `Player ${idx+1}`;
    }

    _applyPlayerNames(){
      if (this.renderer && typeof this.renderer.setPlayerNames === "function"){
        this.renderer.setPlayerNames(this._currentPlayerNames());
      }
    }

    _updateAiUi(){
      if (this.aiStatusEl){
        this.aiStatusEl.hidden = !(this.aiEnabled && !this.onlineActive);
      }
      if (this.aiDifficultyLabel && this.aiEnabled && !this.onlineActive){
        const name=this.aiDifficultyNames[this.aiDepth] || `Depth ${this.aiDepth}`;
        this.aiDifficultyLabel.textContent=name;
      }
    }

    _resetOnlineState(){
      if (this.onlineChannel){
        try{ this.onlineChannel.close(); }catch(_e){}
      }
      if (this.onlinePeer){
        try{ this.onlinePeer.close(); }catch(_e){}
      }
      this.onlineActive=false;
      this.onlineRole=null;
      this.onlinePlayerIndex=null;
      this.onlinePeer=null;
      this.onlineChannel=null;
      this._suspendOnline=false;
      this._currentSeed=null;
      if (this.onlineHostOffer) this.onlineHostOffer.value="";
      if (this.onlineHostAnswer) this.onlineHostAnswer.value="";
      if (this.onlineGuestOffer) this.onlineGuestOffer.value="";
      if (this.onlineGuestAnswer) this.onlineGuestAnswer.value="";
      if (this.onlineStatusEl) this.onlineStatusEl.textContent="Waiting to start...";
      if (this.btnUndo) this.btnUndo.disabled=false;
      if (this.btnNew) this.btnNew.disabled=false;
      const crossInput=document.getElementById("cross-count");
      if (crossInput) crossInput.disabled=false;
      this._refreshControlsForTurn();
    }

    _setMenuSelection(active){
      if (this.menuStartPvpBtn){
        this.menuStartPvpBtn.classList.toggle("primary", active==="pvp");
      }
      if (this.menuStartPvcBtn){
        this.menuStartPvcBtn.classList.toggle("primary", active==="pvc");
      }
      if (this.menuStartOnlineBtn){
        this.menuStartOnlineBtn.classList.toggle("primary", active==="online");
      }
    }

    _showMenuRoot(){
      this._cancelPendingAI();
      this._resetOnlineState();
      this.mode=null;
      this.aiEnabled=false;
      this._updateAiUi();
      this._setSectionVisible(this.gameScreen, false);
      this._setSectionVisible(this.rulesScreen, false);
      this._setSectionVisible(this.onlineScreen, false);
      this._setSectionVisible(this.menuScreen, true);
      if (this.menuRootOptions) this.menuRootOptions.hidden=false;
      if (this.menuDifficulty) this.menuDifficulty.hidden=true;
      if (this.menuBackBtn) this.menuBackBtn.hidden=true;
      this._setMenuSelection(null);
      this._setBoardBlurred(false);
      this._hideGameOver(true);
      this._forceHideRaymondEgg();
      this._applyPlayerNames();
    }

    _showDifficultySelector(){
      this._setSectionVisible(this.menuScreen, true);
      if (this.menuRootOptions) this.menuRootOptions.hidden=true;
      if (this.menuDifficulty) this.menuDifficulty.hidden=false;
      if (this.menuBackBtn) this.menuBackBtn.hidden=false;
      this._setMenuSelection("pvc");
    }

    _showOnlineSetup(){
      this._cancelPendingAI();
      this._resetOnlineState();
      this._setSectionVisible(this.menuScreen, false);
      this._setSectionVisible(this.rulesScreen, false);
      this._setSectionVisible(this.gameScreen, false);
      this._setSectionVisible(this.onlineScreen, true);
      if (this.menuBackBtn) this.menuBackBtn.hidden=true;
      this._setMenuSelection("online");
      if (this.onlineStatusEl) this.onlineStatusEl.textContent = "Waiting to start...";
    }

    _showRulesScreen(){
      this._cancelPendingAI();
      this._setMenuSelection(null);
      this._setSectionVisible(this.menuScreen, false);
      this._setSectionVisible(this.gameScreen, false);
      this._setSectionVisible(this.rulesScreen, true);
    }

    _returnToMenu(){
      this._showMenuRoot();
    }

    _enterGameMode(mode, depth=1){
      this._cancelPendingAI();
      this.mode=mode;
      this.aiEnabled = (mode === "pvc");
      if (this.aiEnabled){
        this.aiDepth=Math.max(1, depth|0);
      }
      this._updateAiUi();
      this._setSectionVisible(this.menuScreen, false);
      this._setSectionVisible(this.rulesScreen, false);
      this._setSectionVisible(this.gameScreen, true);
      if (this.menuRootOptions) this.menuRootOptions.hidden=false;
      if (this.menuDifficulty) this.menuDifficulty.hidden=true;
      if (this.menuBackBtn) this.menuBackBtn.hidden=true;
      this._setMenuSelection(null);
      this._resizeCanvas();
      this.renderer._compute_view_transform();
      this._startNewGame();
      this._queueAIMoveIfNeeded();
    }

    _queueAIMoveIfNeeded(){
      if (!this.aiEnabled) return;
      if (!this._isInMatch()) return;
      if (!this.game || this.game.finished) return;
      if (this._aiThinking) return;
      if (this.game.player !== this.aiPlayerIndex) return;
      this._aiThinking=true;
      this._setControlsDisabled(true);
      this._aiPendingTimer = window.setTimeout(()=>{ this._executeAIMove(); }, 140);
    }

    _executeAIMove(){
      this._aiPendingTimer=null;
      if (!this.aiEnabled || !this._isInMatch() || !this.game || this.game.finished || this.game.player!==this.aiPlayerIndex){
        this._aiThinking=false;
        this._setControlsDisabled(false);
        return;
      }
      let move=null;
      if (typeof this.game.cloneState === "function"){
        const root=this.game.cloneState();
        move=this._chooseAIMove(root, this.aiDepth);
      }
      if (!move){
        const fallbackMoves=this._enumerateMoves(this.game);
        if (fallbackMoves.length){
          move=fallbackMoves[Math.floor(Math.random()*fallbackMoves.length)];
        }
      }
      if (!move){
        this._aiThinking=false;
        this._setControlsDisabled(false);
        return;
      }
      this.undo_stack.push(this.snapshot());
      this.game.pending_cross=[move.nid, move.state];
      const result=this.game.commit_and_score();
      this._finalizeCommit(result, move);
      this._aiThinking=false;
      this._setControlsDisabled(false);
      this._queueAIMoveIfNeeded();
    }

    _chooseAIMove(gameState, depth){
      const moves=this._enumerateMoves(gameState);
      if (!moves.length) return null;
      let bestMove=null;
      let bestScore=-Infinity;
      let alpha=-Infinity, beta=Infinity;
      for (const move of moves){
        const child=gameState.cloneState();
        child.pending_cross=[move.nid, move.state];
        child.commit_and_score();
        const score=this._minimax(child, depth-1, alpha, beta, this.aiPlayerIndex);
        if (score>bestScore){
          bestScore=score;
          bestMove=move;
        }
        alpha=Math.max(alpha, bestScore);
        if (beta<=alpha) break;
      }
      return bestMove;
    }

    _enumerateMoves(game){
      const moves=[];
      if (!game || !game.cross_state) return moves;
      for (const key of Object.keys(game.cross_state)){
        if (game.cross_state[key]==='X'){
          const nid=parseInt(key,10);
          moves.push({nid, state:'A'});
          moves.push({nid, state:'B'});
        }
      }
      return moves;
    }

    _minimax(game, depth, alpha, beta, aiIndex){
      if (depth<=0 || !game || game.finished){
        return this._evaluateGame(game, aiIndex);
      }
      const moves=this._enumerateMoves(game);
      if (!moves.length){
        return this._evaluateGame(game, aiIndex);
      }
      const maximizing = game.player===aiIndex;
      if (maximizing){
        let value=-Infinity;
        for (const move of moves){
          const child=game.cloneState();
          child.pending_cross=[move.nid, move.state];
          child.commit_and_score();
          const score=this._minimax(child, depth-1, alpha, beta, aiIndex);
          value=Math.max(value, score);
          alpha=Math.max(alpha, value);
          if (beta<=alpha) break;
        }
        return value;
      }else{
        let value=Infinity;
        for (const move of moves){
          const child=game.cloneState();
          child.pending_cross=[move.nid, move.state];
          child.commit_and_score();
          const score=this._minimax(child, depth-1, alpha, beta, aiIndex);
          value=Math.min(value, score);
          beta=Math.min(beta, value);
          if (beta<=alpha) break;
        }
        return value;
      }
    }

    _evaluateGame(game, aiIndex){
      if (!game) return 0;
      const opponent=aiIndex^1;
      const scoreDelta=(game.scores?.[aiIndex] ?? 0) - (game.scores?.[opponent] ?? 0);
      const remaining=this._enumerateMoves(game).length/2;
      const turnBias = game.player===aiIndex ? 0.05 : -0.05;
      return scoreDelta - 0.01*remaining + turnBias;
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
      const finished=this.game.finished;
      return {cs, player, scores, claimed, seen, finished};
    }
    restore(snap){
      const {cs, player, scores, claimed, seen, finished} = snap;
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
      this.game.finished=!!finished;
      this.game.pending_cross=null;
    }

    _bindEvents(){
      const startPvp=this.menuStartPvpBtn;
      if (startPvp){
        startPvp.addEventListener("click", ()=>{
          this._enterGameMode("pvp");
        });
      }
      const startPvc=this.menuStartPvcBtn;
      if (startPvc){
        startPvc.addEventListener("click", ()=>{
          this._setMenuSelection("pvc");
          this._showDifficultySelector();
        });
      }
      const startOnline=this.menuStartOnlineBtn;
      if (startOnline){
        startOnline.addEventListener("click", ()=>{
          this._showOnlineSetup();
        });
      }
      const menuRules=document.getElementById("menu-open-rules");
      if (menuRules){
        menuRules.addEventListener("click", ()=>{ this._showRulesScreen(); });
      }
      if (this.menuBackBtn){
        this.menuBackBtn.addEventListener("click", ()=>{ this._showMenuRoot(); });
      }
      if (this.rulesBackBtn){
        this.rulesBackBtn.addEventListener("click", ()=>{ this._showMenuRoot(); });
      }
      if (this.btnReturnMenu){
        this.btnReturnMenu.addEventListener("click", ()=>{ this._returnToMenu(); });
      }
      if (this.onlineHostCreateBtn){
        this.onlineHostCreateBtn.addEventListener("click", ()=>{ this._onlineStartHosting(); });
      }
      if (this.onlineHostApplyBtn){
        this.onlineHostApplyBtn.addEventListener("click", ()=>{ this._onlineApplyHostAnswer(); });
      }
      if (this.onlineGuestConnectBtn){
        this.onlineGuestConnectBtn.addEventListener("click", ()=>{ this._onlineJoinFromOffer(); });
      }
      if (this.onlineCancelBtn){
        this.onlineCancelBtn.addEventListener("click", ()=>{ this._showMenuRoot(); });
      }
      document.querySelectorAll(".difficulty-btn").forEach(btn=>{
        btn.addEventListener("click", ()=>{
          const depth=parseInt(btn.dataset.depth, 10) || 1;
          this._enterGameMode("pvc", depth);
        });
      });

      if (this.btnConfirm){
        this.btnConfirm.addEventListener("click", ()=>{
          if (this._isInteractionLocked()) return;
          this._commitPending();
        });
      }
      if (this.btnNew){
        this.btnNew.addEventListener("click", ()=>{
          if (!this._isInMatch()) return;
          if (this.onlineActive){
            if (this.onlineRole!=="host") return;
            const seed=Math.floor(Math.random()*1e9);
            this._startNewGame(seed, {broadcast:true});
          }else{
            this._startNewGame();
          }
        });
      }
      if (this.btnUndo){
        this.btnUndo.addEventListener("click", ()=>{
          if (!this._isInMatch() || this._aiThinking) return;
          if (this.onlineActive) return;
          if (this.undo_stack.length){
            this.restore(this.undo_stack.pop());
            this._syncGameOverOverlay();
            this._queueAIMoveIfNeeded();
          }
        });
      }
      const btnOverlayNew=document.getElementById("game-over-new");
      if (btnOverlayNew){
        btnOverlayNew.addEventListener("click", ()=>{
          if (this.btnNew){ this.btnNew.click(); }
          else{ this._startNewGame(); }
        });
      }
      const btnOverlayView=document.getElementById("game-over-view");
      if (btnOverlayView){
        btnOverlayView.addEventListener("click", ()=>{
          this._hideGameOver(true);
        });
      }
      const crossInput=document.getElementById("cross-count");
      if (crossInput){
        const apply=()=>{
          if (this.onlineActive) return;
          const prevTarget=this.targetCrossCount;
          const nextTarget=this._readCrossCount();
          if (nextTarget===prevTarget && this.game && !this.game.finished && this.game.nodes){
            return;
          }
          this.targetCrossCount=nextTarget;
          this._startNewGame();
        };
        crossInput.addEventListener("change", apply);
        crossInput.addEventListener("keydown", (ev)=>{
          if (ev.key==="Enter"){
            ev.preventDefault();
            apply();
          }
        });
      }

      this.canvas.addEventListener("mousedown", (ev)=>{
        if (!this._isInMatch()) return;
        if (this._isInteractionLocked()) return;
        if (ev.button!==0 || this.game.finished) return;
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
        this._syncPendingWithOnline();
      });

      window.addEventListener("keydown", (ev)=>{
        if (!this._isInMatch()) return;
        if (ev.key==="Enter" || ev.key==="c"){
          if (this._isInteractionLocked()) return;
          this._commitPending();
        }else if (ev.key==="n"){
          if (this.onlineActive){
            if (this.onlineRole!=="host") return;
            const seed=Math.floor(Math.random()*1e9);
            this._startNewGame(seed, {broadcast:true});
          }else{
            this._startNewGame();
          }
        }else if (ev.key==="u" || ev.key==="Backspace"){
          if (this._aiThinking) return;
          if (this.onlineActive) return;
          if (this.undo_stack.length){
            this.restore(this.undo_stack.pop());
            this._syncGameOverOverlay();
            this._queueAIMoveIfNeeded();
          }
        }
      });
    }

    _commitPending(){
      if (!this._isInMatch()) return;
      if (this._isComputerTurn() || this._aiThinking) return;
      if (this.onlineActive && this.game.player!==this.onlinePlayerIndex) return;
      if (!this.game.pending_cross) return;
      this.undo_stack.push(this.snapshot());
      const move=this.game.pending_cross ? {nid:this.game.pending_cross[0], state:this.game.pending_cross[1]} : null;
      const result=this.game.commit_and_score();
      this._finalizeCommit(result, move);
      this._queueAIMoveIfNeeded();
    }

    _finalizeCommit(result, move=null){
      if (this.onlineActive && !this._suspendOnline && move){
        this._broadcastOnline({type:"commit", move});
      }
      const awards=result && result.awards ? result.awards : [];
      if (awards.length){
        const msg = awards.map(a => {
          const token=a.type==='outside'?'outside':'disk';
          const label=this._playerDisplayName(a.player);
          return `+1 ${token} (${label})`;
        }).join("; ");
        this.renderer.showToast(msg, 1500);
      }
      if (result && result.gameOver){
        this._handleGameOver(result.gameOver, {triggerEgg:true});
        this._cancelPendingAI();
      }else{
        this._clearVictoryDisplay();
      }
      if (this.onlineActive){
        if (result && result.gameOver){
          this._onlineStatus("Game complete.");
        }else if (this.game && this.onlinePlayerIndex!=null){
          this._onlineStatus(this.game.player===this.onlinePlayerIndex ? "Your turn!" : "Opponent's turn...");
        }
        this._refreshControlsForTurn();
      }
    }

    _startNewGame(seedOverride=null, options={}){
      if (!this._isInMatch()) return;
      this._cancelPendingAI();
      const broadcast = options.broadcast ?? false;
      const forcedCount = options.forceCrossCount;
      const crossInput=document.getElementById("cross-count");
      let desired = (forcedCount!=null) ? forcedCount : this._readCrossCount();
      this.targetCrossCount = desired;
      if (forcedCount!=null && crossInput){
        crossInput.value = `${forcedCount}`;
      }
      if (crossInput && this.onlineActive){
        crossInput.disabled = true;
      }
      let seedToUse = seedOverride;
      if (this.onlineActive && seedToUse==null && this.onlineRole==='host'){
        seedToUse = Math.floor(Math.random()*1e9);
      }
      this._currentSeed = seedToUse;
      const game = this._makeGameWithBackoff(seedToUse, this.baseMinCrossSep, this.baseMinCrossAngleDeg);
      const actualCrosses = typeof game.crossCount === "number"
        ? game.crossCount
        : game.nodes.reduce((sum,node)=>sum+(node.type==='cross'?1:0),0);
      if (actualCrosses !== desired){
        this.targetCrossCount = actualCrosses;
        this._setCrossCountInput(actualCrosses);
      }
      this.game = game;
      this.renderer = new Renderer(this.game, this.canvas);
      this._applyPlayerNames();
      this.undo_stack=[];
      this._clearVictoryDisplay();
      if (actualCrosses !== desired){
        this.renderer.showToast(`Using ${actualCrosses} intersections (closest to requested ${desired}).`, 2400);
      }
      this._setControlsDisabled(false);
      this._queueAIMoveIfNeeded();
      if (this.onlineActive){
        if (this.btnUndo) this.btnUndo.disabled=true;
        if (this.btnNew) this.btnNew.disabled = this.onlineRole!=="host";
        if (broadcast && this.onlineRole==='host'){
          this._broadcastOnline({type:"newGame", seed: seedToUse, crossCount:this.targetCrossCount});
        }
        if (this.onlineRole==='host'){
          this._onlineStatus("Connected — You are the host. Your turn!");
        }else{
          this._onlineStatus("Connected — Waiting for host move...");
        }
        this._refreshControlsForTurn();
      }
    }

    _syncGameOverOverlay(){
      if (this.game.finished){
        this._handleGameOver({
          winner:this.game.currentWinner(),
          scores:[...this.game.scores],
          tie:this.game.currentWinner()==null,
        }, {triggerEgg:false});
      }else{
        this._clearVictoryDisplay();
      }
    }

    _broadcastOnline(payload){
      if (!this.onlineActive || !this.onlineChannel || !payload || this._suspendOnline) return;
      try{
        this.onlineChannel.send(JSON.stringify(payload));
      }catch(_err){
        // ignore
      }
    }

    _syncPendingWithOnline(){
      if (!this.onlineActive || this._suspendOnline) return;
      const pending=this.game && this.game.pending_cross ? {nid:this.game.pending_cross[0], state:this.game.pending_cross[1]} : null;
      this._broadcastOnline({type:"pending", pending});
    }

    _handleGameOver(detail, {triggerEgg=false}={}){
      const showCard = () => {
        this._renderGameOverCard(detail);
      };
      if (!triggerEgg || !RAYMOND_EGG_ENABLED){
        if (!this._gameOverPendingPromise) showCard();
        return;
      }
      if (this._gameOverPendingPromise){
        return;
      }
      this._hideGameOver(true);
      this._gameOverPendingPromise = this._playVictoryEgg()
        .catch(()=>{})
        .then(showCard)
        .finally(()=>{
          this._gameOverPendingPromise=null;
        });
    }

    _onlineStatus(text){
      if (this.onlineStatusEl && typeof text === "string"){
        this.onlineStatusEl.textContent=text;
      }
    }

    async _onlineStartHosting(){
      this._resetOnlineState();
      this.onlineRole="host";
      this._onlineStatus("Generating hosting code...");
      try{
        const pc=this._createOnlinePeer("host");
        this.onlinePeer=pc;
        const channel=pc.createDataChannel("splice");
        this._setOnlineChannel(channel, "host");
        const offer=await pc.createOffer();
        await pc.setLocalDescription(offer);
        const desc=await this._waitForIce(pc);
        if (!desc){ throw new Error("No local description available"); }
        const encoded=this._encodeSession(desc);
        if (this.onlineHostOffer) this.onlineHostOffer.value=encoded;
        this._onlineStatus("Share the hosting code, then paste the opponent's answer below.");
      }catch(err){
        this._onlineStatus("Unable to create hosting code. Try again.");
        console.error("online host error", err);
      }
    }

    async _onlineApplyHostAnswer(){
      if (!this.onlinePeer){
        this._onlineStatus("Create a hosting code first.");
        return;
      }
      const answer=this.onlineHostAnswer ? this.onlineHostAnswer.value.trim() : "";
      if (!answer){
        this._onlineStatus("Paste the opponent's answer before applying.");
        return;
      }
      try{
        const desc=this._decodeSession(answer);
        await this.onlinePeer.setRemoteDescription(desc);
        this._onlineStatus("Answer applied. Waiting for the opponent to connect...");
      }catch(err){
        this._onlineStatus("Could not apply answer. Check the code and try again.");
        console.error("apply answer", err);
      }
    }

    async _onlineJoinFromOffer(){
      this._resetOnlineState();
      this.onlineRole="guest";
      const offerText=this.onlineGuestOffer ? this.onlineGuestOffer.value.trim() : "";
      if (!offerText){
        this._onlineStatus("Paste the host's code to join.");
        return;
      }
      try{
        const pc=this._createOnlinePeer("guest");
        this.onlinePeer=pc;
        const remote=this._decodeSession(offerText);
        await pc.setRemoteDescription(remote);
        const answer=await pc.createAnswer();
        await pc.setLocalDescription(answer);
        const desc=await this._waitForIce(pc);
        if (!desc){ throw new Error("No local description available"); }
        if (this.onlineGuestAnswer) this.onlineGuestAnswer.value=this._encodeSession(desc);
        this._onlineStatus("Send your answer to the host, then wait for the connection.");
      }catch(err){
        this._onlineStatus("Unable to join with that code. Please check and try again.");
        console.error("online join error", err);
      }
    }

    _createOnlinePeer(role){
      const pc=new RTCPeerConnection({
        iceServers:[
          {urls:["stun:stun.l.google.com:19302","stun:stun1.l.google.com:19302"]}
        ]
      });
      pc.onconnectionstatechange=()=>{
        if (pc.connectionState==='failed' || pc.connectionState==='disconnected' || pc.connectionState==='closed'){
          this._onlineConnectionClosed();
        }
      };
      pc.oniceconnectionstatechange=()=>{
        if (pc.iceConnectionState==='failed' || pc.iceConnectionState==='disconnected'){
          this._onlineConnectionClosed();
        }
      };
      if (role==='guest'){
        pc.ondatachannel=(ev)=>{
          this._setOnlineChannel(ev.channel, "guest");
        };
      }
      return pc;
    }

    _setOnlineChannel(channel, role){
      if (!channel) return;
      this.onlineChannel=channel;
      channel.onopen=()=>{ this._onOnlineChannelOpen(role); };
      channel.onclose=()=>{ this._onlineConnectionClosed(); };
      channel.onmessage=(ev)=>{
        this._handleOnlineMessage(ev.data);
      };
    }

    async _waitForIce(pc){
      if (pc.iceGatheringState === 'complete' && pc.localDescription){
        return pc.localDescription;
      }
      await new Promise(resolve=>{
        const check=()=>{
          if (pc.iceGatheringState==='complete'){
            pc.removeEventListener('icegatheringstatechange', check);
            resolve();
          }
        };
        pc.addEventListener('icegatheringstatechange', check);
        setTimeout(()=>{
          if (pc.iceGatheringState==='complete'){
            pc.removeEventListener('icegatheringstatechange', check);
            resolve();
          }
        }, 3000);
      });
      return pc.localDescription;
    }

    _encodeSession(desc){
      return btoa(JSON.stringify(desc));
    }

    _decodeSession(str){
      return JSON.parse(atob(str));
    }

    _onOnlineChannelOpen(role){
      this._onlineStatus("Connection established. Syncing game...");
      this._beginOnlinePlay(role);
      if (role==='host'){
        const seed=Math.floor(Math.random()*1e9);
        this._startNewGame(seed, {broadcast:true});
      }else{
        this._onlineStatus("Connected — waiting for host to start a game.");
      }
    }

    _beginOnlinePlay(role){
      this.onlineActive=true;
      this.onlineRole=role;
      this.mode=`online-${role}`;
      this.aiEnabled=false;
      this.onlinePlayerIndex = role==='host' ? 0 : 1;
      this._updateAiUi();
      this._setSectionVisible(this.menuScreen, false);
      this._setSectionVisible(this.rulesScreen, false);
      this._setSectionVisible(this.onlineScreen, false);
      this._setSectionVisible(this.gameScreen, true);
      this._setMenuSelection(null);
      this._resizeCanvas();
      this.renderer._compute_view_transform();
      this._applyPlayerNames();
      const crossInput=document.getElementById("cross-count");
      if (crossInput) crossInput.disabled=true;
      if (this.btnUndo) this.btnUndo.disabled=true;
      if (this.btnNew) this.btnNew.disabled = role!=='host';
      this._refreshControlsForTurn();
    }

    _onlineConnectionClosed(){
      if (!this.onlineActive) return;
      this._onlineStatus("Connection closed. Returning to menu...");
      if (this.renderer && this.renderer.showToast){
        this.renderer.showToast("Online connection closed.", 2200);
      }
      setTimeout(()=>{ this._showMenuRoot(); }, 1400);
    }

    _handleOnlineMessage(raw){
      let msg=null;
      try{ msg=JSON.parse(raw); }
      catch(_err){ return; }
      if (!msg || typeof msg.type!=='string') return;
      if (msg.type==="pending"){
        this._handleOnlinePending(msg.pending);
      }else if (msg.type==="commit"){
        this._handleOnlineCommit(msg.move);
      }else if (msg.type==="newGame"){
        this._handleOnlineNewGame(msg);
      }
    }

    _handleOnlinePending(pending){
      if (!this.onlineActive) return;
      this._suspendOnline=true;
      if (pending && typeof pending.nid === 'number' && pending.state){
        this.game.pending_cross=[pending.nid, pending.state];
      }else{
        this.game.pending_cross=null;
      }
      this._suspendOnline=false;
      this._refreshControlsForTurn();
    }

    _handleOnlineCommit(move){
      if (!this.onlineActive || !move) return;
      this._suspendOnline=true;
      this.game.pending_cross=[move.nid, move.state];
      const result=this.game.commit_and_score();
      this._finalizeCommit(result, move);
      this._suspendOnline=false;
    }

    _handleOnlineNewGame(msg){
      if (!this.onlineActive || !msg) return;
      const count = typeof msg.crossCount === 'number' ? msg.crossCount : null;
      this._suspendOnline=true;
      this._startNewGame(msg.seed ?? null, {broadcast:false, forceCrossCount: count});
      this._suspendOnline=false;
    }

    _clearVictoryDisplay(){
      this._setBoardBlurred(false);
      this._hideGameOver(true);
      this._forceHideRaymondEgg();
    }

    _renderGameOverCard(detail){
      const overlay=document.getElementById("game-over");
      if (!overlay) return;
      this._setBoardBlurred(false);
      const resultEl=document.getElementById("game-over-result");
      const scoreEl=document.getElementById("game-over-score");
      const scores = detail && detail.scores ? detail.scores : this.game.scores;
      if (scoreEl && scores){ scoreEl.textContent = `${scores[0]} — ${scores[1]}`; }
      if (resultEl){
        if (detail && detail.tie){
          resultEl.textContent = "It's a tie!";
        }else{
          const winner = detail && typeof detail.winner === "number" ? detail.winner : this.game.currentWinner();
          const winName = winner==null ? null : this._playerDisplayName(winner);
          resultEl.textContent = winner==null ? "Game complete" : `${winName} wins!`;
        }
      }
      if (this._gameOverHideTimer){
        clearTimeout(this._gameOverHideTimer);
        this._gameOverHideTimer=null;
      }
      overlay.classList.remove("show");
      overlay.hidden=true;
    }

    _hideGameOver(immediate=false){
      const overlay=document.getElementById("game-over");
      if (!overlay) return;
      overlay.classList.remove("show");
      this._setBoardBlurred(false);
      if (this._gameOverHideTimer){
        clearTimeout(this._gameOverHideTimer);
        this._gameOverHideTimer=null;
      }
      if (immediate){
        overlay.hidden=true;
        return;
      }
      this._gameOverHideTimer = setTimeout(()=>{
        overlay.hidden=true;
        this._gameOverHideTimer=null;
      }, 220);
    }

    _playVictoryEgg(){
      const egg=document.getElementById("raymond-egg");
      const img = egg ? egg.querySelector("img") : null;
      if (!egg || !img) return Promise.resolve();
      return new Promise((resolve)=>{
        if (this._raymondHoldTimer){
          clearTimeout(this._raymondHoldTimer);
          this._raymondHoldTimer=null;
        }
        egg.hidden=false;
        egg.classList.add("active");
        img.classList.remove("raymond-animate");
        void img.offsetWidth;
        const finish=()=>{
          if (this._raymondHoldTimer){
            clearTimeout(this._raymondHoldTimer);
            this._raymondHoldTimer=null;
          }
          this._raymondHoldTimer=setTimeout(()=>{
            egg.classList.remove("active");
            img.classList.remove("raymond-animate");
            egg.hidden=true;
            this._raymondHoldTimer=null;
            resolve();
          }, 1000);
        };
        const onAnimEnd=()=>{
          img.removeEventListener("animationend", onAnimEnd);
          finish();
        };
        img.addEventListener("animationend", onAnimEnd, {once:true});
        const fallback=setTimeout(()=>{
          img.removeEventListener("animationend", onAnimEnd);
          finish();
        }, 900);
        const clearFallback=()=>{ clearTimeout(fallback); };
        img.addEventListener("animationend", clearFallback, {once:true});
        img.classList.add("raymond-animate");
      });
    }

    _forceHideRaymondEgg(){
      const egg=document.getElementById("raymond-egg");
      const img = egg ? egg.querySelector("img") : null;
      if (!egg || !img) return;
      if (this._raymondHoldTimer){
        clearTimeout(this._raymondHoldTimer);
        this._raymondHoldTimer=null;
      }
      egg.classList.remove("active");
      img.classList.remove("raymond-animate");
      egg.hidden=true;
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
  const app = new App(null, 2.0, 25.0);
  app.run();
})();
