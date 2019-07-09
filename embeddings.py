import numpy as np

class Embeddings(object):
    def __init__(self):
        pass
    def load_embeddings(self):
        d = {}
        with open('./fast_text.vec') as f:
            for i, line in enumerate(f.readlines()):
                word = line.split()[0]
                vector_string = line.split()[1:]
                d[word] = np.array(vector_string, dtype=np.float32)
        return d
    
    def write_json(self, file_name, data):
        with open(file_name + '.json', 'w') as f:
            json.dump(data, f)
            
    def main(self):
        d = self.load_embeddings()
        
        
        
    if __name__ == '__main__':
        main()