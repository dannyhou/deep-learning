#coding:utf-8
from functools import reduce
## 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算。
class Node(object):
    def __int__(self,layer_index,node_index):
        '''
        构造节点对象。
        layer_index: 节点所属的层的编号
        node_index: 节点的编号
        '''
        self.layer_index=layer_index
        self.node_index=node_index
        self.downstream=[]
        self.upstream=[]
        self.output=0
        self.delta=0

        def set_output(self,output):
            '''
            :param self:
            :param output:
            :return:None
            设置节点输出，如果节点属于输入层，则调用这个函数
            '''
            self.output=output

        def append_downstream_connection(sef,conn):
            '''
            :param sef:
            :param conn:
            :return:None
             添加一个到下游节点的链接
            '''
            self.downstream.append(conn)


        def append_upstream_connection(self,conn):
            '''
            :param self:
            :param conn:
            :return:None
             添加一个到上游节点的链接
            '''
            self.upstream.append(conn)

        def calc_output(self):
            '''
            :param self:
            :return:
            根据公式计算几点输出：y=sigmoid(w*x)
            '''
            output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
            self.output=sigmoid(output)

        def calc_hidden_layer_delta(self):
             '''
             :param self:
             :param label:
             :return:
              节点属于隐藏层是，根据公式算delta=a_i(1-a_i)*∑w_kiδ
             '''
             downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,self.downstream, 0.0)
             self.delta = self.output * (1 - self.output) * downstream_delta
        def calc_output_layer_delta(self,label):
            '''
            :param self:
            :param label:
            :return:
            节点属于输出层时，根据式公式计算delta=y(1-y)(Y-y)
            '''
            self.delta = self.output * (1 - self.output) * (label - self.output)


        def __str__(self):
            '''
            打印节点的信息
            '''
            node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
            downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
            upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
            return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str
