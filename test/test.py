import simpai


@simpai.hyperparam.setHp('loss')
def addint(a, b):
    print(f'a is {a}, b is {b}')
    return a + b

if __name__ == '__main__':
    simpai.hyperparam.setHp('epoch_num', 100)
    print(simpai.hyperparam.getHp('loss')(3, 4))
    print(simpai.hyperparam.getHp('epoch_num'))
