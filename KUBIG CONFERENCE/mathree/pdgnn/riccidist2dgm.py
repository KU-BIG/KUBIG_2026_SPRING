import numpy as np
import networkx as nx
import gudhi
import time
import sg2dgm.PersistenceImager as pimg
from sg2dgm.dgformat import Diagram, _Point, flip_dgm, print_dgm
from multiprocessing.dummy import Pool as ThreadPool
import sys
from sg2dgm.accelerated_PD import perturb_filter_function, Union_find, Accelerate_PD


def _gudhi_persistence(simplices, ascending=True):
    """
    gudhi SimplexTree로 0D/1D persistence 계산.
    ascending=False 이면 descending filtration (값을 뒤집어서 계산 후 복원).
    반환: [Diagram_dim0, Diagram_dim1]
    """
    st = gudhi.SimplexTree()
    sign = 1 if ascending else -1
    for simplex, val in simplices:
        st.insert(simplex, filtration=sign * val)

    st.compute_persistence()

    dgms = [Diagram(), Diagram()]
    for dim, (birth, death) in st.persistence():
        if dim >= 2:
            continue
        # inf 제거
        if birth == float('inf') or death == float('inf'):
            continue
        if ascending:
            dgms[dim].append(_Point(birth, death))
        else:
            # 값을 뒤집었으므로 복원: birth_orig = -death_gudhi
            dgms[dim].append(_Point(-death, -birth))

    return dgms
# ────────────────────────────────────────────────────────────────────────────


class filtration():
    def __init__(self, g, u, v, hop, ricci_curv):
        self.g = g
        self.n = len(g)
        self.root_1 = u
        self.root_2 = v
        self.hop = hop
        self.ricci_curv = ricci_curv

    def build_fv(self, weight_graph=False, norm=True):
        for x in self.g.nodes():
            if x in [self.root_1, self.root_2]:
                self.g.nodes[x]['max'] = 0
                self.g.nodes[x]['min'] = 0
                self.g.nodes[x]['sum'] = 0
            else:
                if weight_graph:
                    try:
                        path_1 = nx.dijkstra_path(self.g, x, self.root_1, weight='weight')
                        dist_1 = sum([self.ricci_curv[(path_1[y], path_1[y + 1])] + 1
                                      for y in range(len(path_1) - 1)])
                    except BaseException:
                        dist_1 = 100
                    try:
                        path_2 = nx.dijkstra_path(self.g, x, self.root_2, weight='weight')
                        dist_2 = sum([self.ricci_curv[(path_2[y], path_2[y + 1])] + 1
                                      for y in range(len(path_2) - 1)])
                    except BaseException:
                        dist_2 = 100
                else:
                    try:
                        dist_1 = nx.shortest_path_length(self.g, x, self.root_1)
                    except BaseException:
                        dist_1 = 100
                    try:
                        dist_2 = nx.shortest_path_length(self.g, x, self.root_2)
                    except BaseException:
                        dist_2 = 100
                self.g.nodes[x]['min'] = min(dist_1, dist_2)
                self.g.nodes[x]['max'] = max(dist_1, dist_2)
                self.g.nodes[x]['sum'] = dist_1 + dist_2
        if norm:
            norm_scaler = float(max([self.g.nodes[x]['max'] for x in self.g.nodes()]))
            norm_scaler_sum = float(max([self.g.nodes[x]['sum'] for x in self.g.nodes()]))
            for x in self.g.nodes():
                self.g.nodes[x]['min'] /= norm_scaler
                self.g.nodes[x]['max'] /= norm_scaler
                self.g.nodes[x]['sum'] /= norm_scaler_sum
        for u, v in self.g.edges():
            self.g[u][v]['min'] = max(self.g.nodes[u]['min'], self.g.nodes[v]['min'])
            self.g[u][v]['max'] = max(self.g.nodes[u]['max'], self.g.nodes[v]['max'])
            self.g[u][v]['sum'] = max(self.g.nodes[u]['sum'], self.g.nodes[v]['sum'])
        return self.g


class graph2dgm():
    def __init__(self, g, **kwargs):
        self.graph = nx.convert_node_labels_to_integers(g)

    def get_simplices(self, gi, key='min'):
        assert str(type(gi)) == "<class 'networkx.classes.graph.Graph'>" \
               or "<class 'networkx.classes.graphviews.SubGraph'>"
        assert len(gi) > 0
        assert len(gi) == max(list(gi.nodes())) + 1
        simplices = []
        for u, v, data in sorted(gi.edges(data=True), key=lambda x: x[2][key]):
            simplices.append(([u, v], data[key]))
        for v, data in sorted(gi.nodes(data=True), key=lambda x: x[1][key]):
            simplices.append(([v], data[key]))
        return simplices

    def get_desc_simplices(self, gi, key='min'):
        assert str(type(gi)) == "<class 'networkx.classes.graph.Graph'>" \
               or "<class 'networkx.classes.graphviews.SubGraph'>"
        assert len(gi) > 0
        assert len(gi) == max(list(gi.nodes())) + 1
        simplices = []
        values = nx.get_node_attributes(gi, key)
        for u, v, data in sorted(gi.edges(data=True), key=lambda x: x[2][key]):
            simplices.append(([u, v], min(values[u], values[v])))
        for v, data in sorted(gi.nodes(data=True), key=lambda x: x[1][key]):
            simplices.append(([v], data[key]))
        return simplices

    def del_inf(self, dgms):
        """inf 포인트 제거 후 Diagram 반환 (gudhi 버전에서는 이미 필터됨)"""
        dgms_list = [Diagram(), Diagram()]
        for i in range(min(2, len(dgms))):
            for pt in dgms[i]:
                b, d_ = pt.birth, pt.death
                if b == float('inf') or d_ == float('inf'):
                    continue
                if b == float('-inf') or d_ == float('-inf'):
                    continue
                dgms_list[i].append(_Point(b, d_))
        return dgms_list

    def compute_PD(self, simplices, sub=True, inf_flag='False'):
        dgms = _gudhi_persistence(simplices, ascending=sub)

        # inf 제거 (inf_flag == 'False' 일 때 — 원본과 동일 로직)
        if inf_flag == 'False':
            dgms = self.del_inf(dgms)

        # 빈 경우 fallback
        if len(dgms[0]) == 0 and len(dgms[1]) == 0:
            return [Diagram([(0, 0)]), Diagram()]
        return dgms

    def get_diagram(self, g, key='min', one_homology_flag=False):
        g = nx.convert_node_labels_to_integers(g)

        if one_homology_flag:
            epd_dgm = self.epd(g, key=key, pd_flag=False)[1]
            epd_dgm = self.post_process(epd_dgm)
            dgms = [[pt.birth, pt.death] for pt in epd_dgm]
            return dgms

        simplices = self.get_simplices(g, key=key)
        down_simplices = self.get_desc_simplices(g, key=key)
        super_dgms = self.compute_PD(down_simplices, sub=False)
        sub_dgms = self.compute_PD(simplices, sub=True)

        _min = min([g.nodes[n][key] for n in g.nodes()])
        _max = max([g.nodes[n][key] for n in g.nodes()]) + 1e-5

        p_min = Diagram([(_min, _max)])
        p_max = Diagram([(_max, _min)])

        sub_dgms[0].append(p_min[0])
        super_dgms[0].append(p_max[0])

        dgms = ([[pt.birth, pt.death] for pt in sub_dgms[0]] +
                [[pt.birth, pt.death] for pt in super_dgms[0]])
        return dgms

    def epd(self, g__, key='min', pd_flag=False, debug_flag=False):
        """
        Extended Persistence Diagram (EPD).
        gudhi.SimplexTree.extend_filtration() 으로 구현.
        """
        g__ = nx.convert_node_labels_to_integers(g__)
        values = nx.get_node_attributes(g__, key)

        st = gudhi.SimplexTree()
        for node in g__.nodes():
            st.insert([node], filtration=values[node])
        for u, v in g__.edges():
            st.insert([u, v], filtration=max(values[u], values[v]))

        st.extend_filtration()
        ext_dgms = st.extended_persistence()
        # ext_dgms = [Ord0, Rel1, Ext0+, Ext1+]  각 항목은 (dim, (birth, death)) 리스트

        # 0D ordinary + 1D extended 만 뽑아서 dionysus Diagram 형식으로 변환
        dgm0 = Diagram()
        dgm1 = Diagram()

        for dim, (b, d_) in ext_dgms[0]:   # Ord0
            if b != float('inf') and d_ != float('inf'):
                dgm0.append(_Point(b, d_))
        for dim, (b, d_) in ext_dgms[2]:   # Ext0
            if b != float('inf') and d_ != float('inf'):
                dgm0.append(_Point(b, d_))

        if not pd_flag:
            for dim, (b, d_) in ext_dgms[3]:  # Ext1
                if b != float('inf') and d_ != float('inf'):
                    dgm1.append(_Point(b, d_))

        dgms = [dgm0, dgm1]

        if debug_flag:
            print('EPD computed via gudhi. dim0 pts:', len(dgm0), 'dim1 pts:', len(dgm1))
        return dgms

    def post_process(self, dgm, debug_flag=False):
        if len(dgm) == 0:
            return Diagram([(0, 0)])
        pts = []
        for p in dgm:
            b = 0 if p.birth == float('-inf') or p.birth == float('inf') else p.birth
            d_ = 0 if p.death == float('inf') or p.death == float('-inf') else p.death
            pts.append(_Point(b, d_))
        result = Diagram(pts)
        if debug_flag:
            print('post_process result:', result)
        result = flip_dgm(result)
        return result


class graph2pi():
    def __init__(self, g, ricci_curv):
        self.graph = nx.convert_node_labels_to_integers(g, label_attribute="old_label")
        self.dict_node = {}
        for new_label in self.graph._node:
            self.dict_node[self.graph._node[new_label]['old_label']] = new_label
        self.ricci_curv = {}
        for i in ricci_curv:
            self.ricci_curv[(self.dict_node[i[0]], self.dict_node[i[1]])] = i[2]
            self.ricci_curv[(self.dict_node[i[1]], self.dict_node[i[0]])] = i[2]
            self.graph[self.dict_node[i[0]]][self.dict_node[i[1]]]['weight'] = i[2] + 1
            self.graph[self.dict_node[i[1]]][self.dict_node[i[0]]]['weight'] = i[2] + 1

    def multi_wrapper(self, args):
        return self.sg2pimg(*args)

    def sg2pimg(self, u, v, hop, weight_graph=False, norm=True,
                extended_flag=False, range='intersection', resolution=5,
                descriptor="min"):
        if range == 'union':
            root = u
            nodes = [root] + [x for u, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            root = v
            nodes = nodes + [root] + [x for v, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            subgraph = self.graph.subgraph(nodes)
            fil = filtration(subgraph, u, v, hop, ricci_curv=self.ricci_curv)
            g = fil.build_fv(weight_graph=weight_graph, norm=norm)
            x = graph2dgm(g)
            diagram_zero = x.get_diagram(g, key=descriptor, one_homology_flag=False)
            diagram_one = x.get_diagram(g, key=descriptor, one_homology_flag=True) if extended_flag else []
            pers_imager = pimg.PersistenceImager(resolution=resolution)
            pers_img = pers_imager.transform(np.array(diagram_zero + diagram_one))
            return pers_img

        elif range == 'intersection':
            root = u
            nodes_u = [root] + [x for u, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            root = v
            nodes_v = [root] + [x for v, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            nodes = list(set(nodes_u) & set(nodes_v))
            subgraph = self.graph.subgraph(nodes)
            fil = filtration(subgraph, u, v, hop, ricci_curv=self.ricci_curv)
            g = fil.build_fv(weight_graph=weight_graph, norm=norm)
            x = graph2dgm(g)
            if descriptor != 'ricci':
                diagram_zero = x.get_diagram(g, key=descriptor, one_homology_flag=False)
                diagram_one = x.get_diagram(g, key=descriptor, one_homology_flag=True) if extended_flag else []
                pers_imager = pimg.PersistenceImager(resolution=resolution)
                pers_img = pers_imager.transform(np.array(diagram_zero + diagram_one))
                return pers_img

        elif range == 'removeinter':
            root = u
            nodes_u = [root] + [x for u, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            root = v
            nodes_v = [root] + [x for v, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
            nodes_union = nodes_u + nodes_v
            nodes_intersec = set(nodes_u) & set(nodes_v)
            nodes = list(set(nodes_union).difference(nodes_intersec)) + [u, v]
            subgraph = self.graph.subgraph(nodes)
            fil = filtration(subgraph, u, v, hop, ricci_curv=self.ricci_curv)
            g = fil.build_fv(weight_graph=weight_graph, norm=norm)
            x = graph2dgm(g)
            diagram_zero = x.get_diagram(g, key=descriptor, one_homology_flag=False)
            diagram_one = x.get_diagram(g, key=descriptor, one_homology_flag=True) if extended_flag else []
            pers_imager = pimg.PersistenceImager(resolution=resolution)
            pers_img = pers_imager.transform(np.array(diagram_zero + diagram_one))
            return pers_img

        else:
            print("Error: 'range' should be 'union' or 'intersection'!")
            sys.exit()

    def sg2dgm_accelerate(self, u, v, hop, extended_flag=False,
                           descriptor="seal", resolution=5, norm=False, cnt=0):
        root = u
        nodes_u = [root] + [x for u, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
        root = v
        nodes_v = [root] + [x for v, x in nx.bfs_edges(self.graph, root, depth_limit=hop)]
        nodes = list(set(nodes_u) & set(nodes_v))
        subgraph = self.graph.subgraph(nodes)
        assert len([cc for cc in nx.connected_components(subgraph)]) == 1
        fil = filtration(subgraph, u, v, hop, ricci_curv=self.ricci_curv)
        g = fil.build_fv(weight_graph=True, norm=norm)
        simplex_filter = perturb_filter_function(g, descriptor=descriptor)
        PD_zero, Pos_edges, Neg_edges = Union_find(simplex_filter)
        PD_one = Accelerate_PD(Pos_edges, Neg_edges, simplex_filter) if extended_flag else []
        pers_imager = pimg.PersistenceImager(resolution=resolution)
        pers_img = pers_imager.transform(np.array(PD_zero + PD_one))
        return pers_img

    def get_pimg(self, cores, hop, weight_graph=False, norm=True,
                 extended_flag=False, range='intersection', resolution=5):
        params = [(u, v, hop, weight_graph, norm, extended_flag, range, resolution)
                  for u, v in self.graph.edges()]
        pool = ThreadPool(cores)
        pool.map(self.multi_wrapper, params)
        pool.close()
        pool.join()

    def get_pimg_for_one_edge(self, u, v, hop=2, norm=True,
                               extended_flag=False, resolution=5,
                               descriptor='min', cnt=0):
        if cnt % 1000 == 0:
            print(f"having computed: {cnt} edges, really computed: {self.cnt_compute} edges, "
                  f"cost {time.time() - self.t1:.1f}s")
        try:
            self.pi_sg[cnt] = self.sg2dgm_accelerate(
                self.dict_node[u], self.dict_node[v], hop,
                norm=True, extended_flag=extended_flag,
                resolution=resolution, descriptor=descriptor
            ).reshape(-1)
            self.cnt_compute += 1
            return self.pi_sg[cnt]
        except BaseException:
            return np.zeros([resolution * resolution])

    def multi_wrapper_all_edges(self, args):
        return self.get_pimg_for_one_edge(*args)

    def get_pimg_for_all_edges(self, total_edges, cores, hop=2, norm=True,
                                extended_flag=False, resolution=5, descriptor='min'):
        self.pi_sg = np.zeros((len(total_edges), resolution * resolution))
        self.cnt_compute = 0
        self.t1 = time.time()
        params = [(edge[0], edge[1], hop, norm, extended_flag, resolution, descriptor, cnt)
                  for cnt, edge in enumerate(total_edges)]
        pool = ThreadPool(cores)
        pool.map(self.multi_wrapper_all_edges, params)
        pool.close()
        pool.join()
