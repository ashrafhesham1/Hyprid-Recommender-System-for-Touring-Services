import sys
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(script_directory,'./data'))
sys.path.append(os.path.join(script_directory,'./model'))


from Model import Model

if __name__ == '__main__':
    model = Model()
    model.fit()

    u,i = '31447','553171'
    print(model.predict(u, i))

    new_u = {553171:4,459804:3,308825:5}
    print(model.predict(new_u, i, k = 200))

    print(model.get_top_N(u, [553171,459804,308825]))