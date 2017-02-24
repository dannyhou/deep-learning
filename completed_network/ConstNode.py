#coding:utf-8
#ConstNode对象，为了实现一个输出恒为1的节点(计算偏置项Wb时需要)
from functools import reduce
class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''
        构造节点对象。
        layer_index: 节点所属的层的编号
        node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output =1
    def append_downstream_connection(self,conn):
        '''
        添加一个到下游节点的链接
        :param conn:
        :return:
        '''
        self.downstream.append(conn)
    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str
