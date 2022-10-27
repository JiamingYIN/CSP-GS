from pathlib import Path
from bs4 import BeautifulSoup as bs
from geopy.distance import great_circle
from collections import defaultdict
from utils import *


class DataLoader(object):
    """
    Load route data from openstreetmap.
    """
    def __init__(self,
                 dir_path: str = None,
                 interval: float = 30):
        """
        nodes: array, [num_nodes, 2], 2个维度分别为lat, lon
        edges: array, [num_edges, 3], [from, to, distance]
        """
        file = Path(os.path.join(dir_path, 'edges.csv'))
        if_exist = file.exists()
        self.dir_path = dir_path
        self.nodes, self.edges = self.load_data(if_exist)
        self.V = self.nodes.shape[0]
        self.E = self.edges.shape[0]
        self.interval = interval

        if os.path.exists(os.path.join(dir_path, 'travel_time.csv')):
            tts = pd.read_csv(os.path.join(dir_path, 'travel_time.csv'), index_col=0)
            self.travel_time = tts.values
            self.G = self.get_graph()
        
        if os.path.exists(os.path.join(dir_path, 'train.csv')):
            train_data = pd.read_csv(os.path.join(dir_path, 'train.csv'), index_col=0)
            self.sources = list(train_data['source'].astype(int))
            self.targets = list(train_data['target'].astype(int))

            # Load Constraints
            Rs = list(train_data['R'])
            # Generate new constraints according to the tightness
            self.Rs = Rs
            self.opt_travel_time = list(train_data['opt_travel_time'])

        # Load Dynamic Graph
        if os.path.exists(os.path.join(dir_path, 'tt_seq.csv')):
            # self.tt_seq = pd.read_csv(os.path.join(dir_path, 'tt_seq.csv'), index_col=0).values[:, :1000]
            self.tt_seq = pd.read_csv(os.path.join(dir_path, 'tt_seq.csv'), index_col=0).values
            self.get_graph_seq()

        if os.path.exists(os.path.join(dir_path, 'highway.csv')):
            self.highway = pd.read_csv(os.path.join(self.dir_path, 'highway.csv'))
            self.raw_edges = pd.read_csv(os.path.join(self.dir_path, 'raw_edges.csv'))
            self.raw_nodes = pd.read_csv(os.path.join(self.dir_path, 'raw_nodes.csv'))

        if os.path.exists(os.path.join(dir_path, 'broken_edges.csv')):
            self.broken_edges = pd.read_csv(os.path.join(self.dir_path, 'broken_edges.csv'), index_col=0).values

    def transform(self, paths):
        paths_list = []
        for path in paths:
            one_path = [int(p.strip()) for p in path[1:-1].split(',')]
            paths_list.append(one_path)
        return paths_list

    def get_opt_path(self):
        train_data = pd.read_csv(os.path.join(self.dir_path, 'train.csv'), index_col=0)
        opt_paths = list(train_data['opt_path'])
        opt_paths_list = self.transform(opt_paths)
        return opt_paths_list

    def load_data(self, if_exist):
        if if_exist:
            nodes = pd.read_csv(os.path.join(self.dir_path, 'nodes.csv'))
            edges = pd.read_csv(os.path.join(self.dir_path, 'edges.csv'))
        else:
            raw_data = bs(open(os.path.join(self.dir_path, 'map.osm'), encoding='utf-8'), 'xml')
            raw_nodes = self.get_nodes(raw_data)
            raw_edges = self.get_edges(raw_nodes, raw_data)
            # 更新index, 去除多余节点
            nodes, edges = self.update(raw_nodes, raw_edges)

        nodes = nodes[['lat', 'lon']].values
        edges = edges[['from', 'to', 'cost']].values
        return nodes, edges

    def get_node_id_map(self):
        nodes = pd.read_csv(os.path.join(self.dir_path, 'nodes.csv'))
        origin_id = nodes['origin_id']
        id = nodes['id']
        idmap = dict(zip(id, origin_id))
        return idmap


    def get_dist_dict(self):
        dist_dict = load_dict(os.path.join(self.dir_path, 'dist.txt'))
        return dist_dict

    def get_nodes(self, raw_data):
        nodes = raw_data.find_all('node')
        datalist = []
        for node in nodes:
            datalist.append([node['id'], node['lat'], node['lon']])
        df = pd.DataFrame(datalist, columns=['origin_id', 'lat', 'lon'])
        df.to_csv(os.path.join(self.dir_path, 'raw_nodes.csv'))
        return df

    def get_edges(self, raw_nodes, raw_data):
        ways = raw_data.find_all('way')
        way_kinds = ['motorway', 'motorway_link', 'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary',
                     'tertiary_link', 'residential', 'living_street', 'service', 'trunk', 'trunk_link', 'unclassified',
                     'road', 'track']

        segMap = defaultdict(int)
        highways = []
        for w in ways:
            flag = False
            way_id = int(w.attrs['id'])
            isOneway = False
            category = 'unclassified'
            for tag in w.find_all('tag'):
                if tag['k'] == 'highway' and tag['v'] in way_kinds:
                    flag = True
                    category = tag['v']
                if tag['k'] == 'oneway' and tag['v'] == 'yes':
                    isOneway = True

            nodes = [nd['ref'] for nd in w.find_all('nd')]
            if flag:
                highways.append([way_id, isOneway, category, nodes])
                for node in nodes:
                    segMap[node] += 1

        df = pd.DataFrame(highways, columns=['id', 'oneway', 'category', 'nodes'])

        junctions = set()
        for k in segMap.keys():
            if segMap[k] > 1:
                junctions.add(k)

        final_list = []
        for index, row in df.iterrows():
            st = 0
            node_list = []
            for i in range(1, len(row['nodes']) - 1):
                if row['nodes'][i] in junctions:
                    node_list.append(row['nodes'][st:i+1])
                    node_list.append(list(reversed(row['nodes'][st:i+1])))
                    st = i
            node_list.append(row['nodes'][st:len(row['nodes'])])
            node_list.append(list(reversed(row['nodes'][st:len(row['nodes'])])))
            for nst in node_list:
                final_list.append([1, row['oneway'], row['category'], nst])
        df = pd.DataFrame(final_list, columns=['id', 'oneway', 'category', 'nodes'])
        df['id'] = df.index
        df['nodes'] = df['nodes'].apply(lambda x: list(x))
        df.to_csv(os.path.join(self.dir_path, 'highway.csv'), index=False)

        # 根据节点的ID获得节点的经纬度
        def get_node_lat_lon(node_id, df):
            node = df[df.origin_id == node_id]
            return node.iloc[0, 1], node.iloc[0, 2]

        # 计算节点间距离
        def cal_distance(nodeA, nodeB):
            latA, lonA = get_node_lat_lon(nodeA, raw_nodes)
            latB, lonB = get_node_lat_lon(nodeB, raw_nodes)
            a = (latA, lonA)
            b = (latB, lonB)
            # distance = vincenty(a, b).m
            dist = great_circle(a, b).m
            return dist

        # 计算GPS点序列的长度
        def distance(idlist):
            ids = [node_id for node_id in idlist]
            dist = 0
            for i in range(len(ids) - 1):
                dist += cal_distance(ids[i], ids[i + 1])
            return dist

        # 生成边数据, 格式为[id, nodes, from, to, cost]
        df['from'] = df['nodes'].apply(lambda x: x[0])
        df['to'] = df['nodes'].apply(lambda x: x[-1])
        df['cost'] = df['nodes'].apply(lambda x: distance(x))
        df = df[['id', 'from', 'to', 'cost', 'nodes']]
        df.to_csv(os.path.join(self.dir_path, 'raw_edges.csv'), index=False)
        return df

    def update(self, raw_nodes, raw_edges):
        nd_from = set(raw_edges['from'].unique())
        nd_to = set(raw_edges['to'].unique())
        nodes = list(nd_from | nd_to)
        nodes = raw_nodes[raw_nodes.origin_id.isin(nodes)]
        id_dict = dict(zip(nodes['origin_id'], range(nodes.shape[0])))
        nodes['id'] = nodes['origin_id'].apply(lambda x: id_dict[x])
        edges = raw_edges.copy()
        edges['from'] = edges['from'].apply(lambda x: id_dict[x])
        edges['to'] = edges['to'].apply(lambda x: id_dict[x])
        nodes.to_csv(os.path.join(self.dir_path, 'nodes.csv'), index=False)
        edges.to_csv(os.path.join(self.dir_path, 'edges.csv'), index=False)
        return nodes, edges

    def get_graph(self):
        """
        返回networkX的Graph对象
        :param edges: array, [num_edges, 3], [from, to, distance]
        :return: nx.Graph
        """
        G = nx.DiGraph()
        for i in range(self.V):
            G.add_node(int(i), lat=self.nodes[i][0], lon=self.nodes[i][1])
        for i in range(self.edges.shape[0]):
            G.add_edge(int(self.edges[i][0]), int(self.edges[i][1]), length=self.edges[i][2], travel_time=self.travel_time[i][2])
        return G

    def get_graph_t(self, time_idx):
        G = self.G.copy()
        for i in range(self.edges.shape[0]):
            from_v, to_v, _ = self.edges[i]
            G[from_v][to_v]['travel_time'] = self.tt_seq[i][time_idx]
        return G

    def get_graph_seq(self): 
        """
        Return: List of networkx.DiGraph
        """
        # self.graph_seq = [self.G]
        self.graph_seq = []
        for t in range(self.tt_seq.shape[1]):
            G = self.get_graph_t(t)
            self.graph_seq.append(G)


if __name__ == '__main__':
    data_loader = DataLoader(dir_path='../koln_data/koln1630_1.5')

    print("Number of Vertices:", data_loader.V)
    print("Number of Edges:   ", data_loader.E)
    print(data_loader.tt_seq.shape)
