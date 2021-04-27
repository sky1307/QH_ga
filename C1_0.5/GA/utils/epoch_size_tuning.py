'''
    Input: so luong model, size epoch min, buoc nhay epoch = 50
    Output: list epoch tuong ung voi cac model con 
'''

def get_epoch_size_list(num_model, epoch_min, epoch_step=50):
    lst_epoch_size  = []
    for i in range(epoch_min, epoch_min + num_model * epoch_step, epoch_step ):
        lst_epoch_size.append(i)
    return lst_epoch_size

if __name__=="__main__":
    lst_model = get_epoch_size_list(4, 100)
    print(lst_model)
