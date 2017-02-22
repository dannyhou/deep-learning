from Perceptron import Perceptron


'''
定义激活函数
'''
def f(x):
	return x
class LinearUnit(Perceptron):
	"""docstring for LinearUnit"""
	def __init__(self, input_num):
		'''初始化线性单元，设置输入参数的个数'''
		Perceptron.__init__(self,input_num,f)
		



if __name__ == '__main__':
	def get_training_dataset():
	    input_vecs = [[5], [3], [8], [1.4], [10.1]]
	    # 期望的输出列表，月薪，注意要与输入一一对应
	    labels = [5500, 2300, 7600, 1800, 11400]
	    return input_vecs, labels
	def train_linear_unit():
		#传入参数特征为1
		lu=LinearUnit(1)
		input_vecs,labels=get_training_dataset()
		#训练数据，迭代10次，学习率为0.1
		lu.train(input_vecs,labels,10,0.1)
		return lu
	linear_unit=train_linear_unit()
	print(linear_unit)
	print(linear_unit.predict([3.4]))
	print(linear_unit.predict([15]))
	print(linear_unit.predict([1.5]))
	print(linear_unit.predict([6.3]))
