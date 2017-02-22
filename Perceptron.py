class Perceptron(object):
    
    def __init__(self,input_num,activator):
        '''
        初始化感知器，设置传入参数个数和激活函数
        激活函数的类型为:double
        '''
        self.activator=activator
        #权重向量初始化，全部置零
        self.weights=[0 for _ in range(input_num)]
        #偏置项初始化
        self.bias=0.0
    
    def __str__(self):
        '''
        打印学习到的权重和偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)
    
    def predict(self,input_vec):
        '''
        输入向量，输出感知结果
        把input_vec[x1,x2,x3.....]和weights[w1,w2,w3....]打包到一起变成[(x1,w1),(x2,w2),(x3,w3)......]
        然后利用map函数计算[x1*w1,x2*w2,x3*w3....]
        最后用reduce求和
        '''
        
#         return self.activator(reduce(lambda a, b: a + b,map(lambda (x, w): x * w,zip(input_vec, self.weights)), 0.0) + self.bias)
        def f1(input_vec,weights):
            sum_=[]
            for x,w in zip(input_vec,weights):
                sum_.append(x*w)
            return sum_
        
        return self.activator(sum(f1(input_vec,self.weights))*1.0+self.bias)
    
    def train(self,input_vecs,labels,iteration,rate):
        '''
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数和学习率
        '''
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)
    
    def _one_iteration(self,input_vecs,labels,rate):
        '''
        一次迭代，把所有的训练数据过一遍
        '''
        ##把输入和输出打包成样本列表[(input_vec,label).......]
        ##每个训练样本是(input_vec,label)
        samples=zip(input_vecs,labels)
        
        #对每个样本按感知规则更新权重
        for (input_vec,label) in samples:
            # 计算感知器在当前权重下的输出
            output=self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec,output,label,rate)
        
    def _update_weights(self,input_vec,output,label,rate):
        '''
        按照感知器规则更新权重
        '''
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta=label-output
        def w(input_vec,weigths,rate,delta):
            w_l=[]
            for x,w in zip(input_vec, self.weights):
               w_l.append(w + rate * delta * x) 
            return w_l
        self.weights=w(input_vec, self.weights,rate,delta)
        # 更新bias
        self.bias += rate * delta

        
        
        
def f(x):
    '''
    定义激活函数
    '''
    return 1 if x>0 else 0

def get_training_dataset():
    '''
    基于and真值表来创建训练数据
    '''
    input_vecs=[[1,1],[1,0],[0,1],[0,0]]
    labels=[1,0,0,0]
    return input_vecs,labels

def train_perceptron():
    '''
    训练感知器
    '''
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p=Perceptron(2,f)
    input_vecs,labels=get_training_dataset()
    #迭代10次，学习率为0.1
    p.train(input_vecs,labels,10,0.1)
    return p

and_perceptron=train_perceptron()
# 打印训练获得的权重
print(and_perceptron)

print(and_perceptron.predict([1,1]))
print(and_perceptron.predict([1,0]))
print(and_perceptron.predict([0,1]))
print(and_perceptron.predict([0,0]))        
