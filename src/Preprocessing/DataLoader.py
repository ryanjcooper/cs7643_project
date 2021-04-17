import numpy as np

class DataLoader():
    def __init__(self, data_dir = '../data/roastme6k'):
        self.data_dir = data_dir
    
    def load(self, remove_deleted_comments=True):
        data = {}

        with open(self.data_dir + '/token.txt', 'r') as f:
            for line in f.readlines():
                fname = line.split('#')[0]
                comment = ' '.join(line.split('\t')[1:]).strip()
                
                if remove_deleted_comments and comment.strip() == 'deleted':
                    continue
                
                try:
                    data[fname]['text'].append(comment)
                except:
                    data.update({
                        fname : {
                            'img' : np.load(self.data_dir + '/image_data/' + fname + '.npy'),
                            'text' : [comment]
                        }
                    })
        return data
