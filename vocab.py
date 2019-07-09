class Vocab:
    def __init__(self, counter, min_freq=None, max_freq=None):
        self.sos = "<sos>"
        self.eos = "<eos>"
        self.pad = "<pad>"
        self.unk = "<unk>"
        
        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3
        
        self._token2idx = {
            self.sos: self.sos_idx,
            self.eos: self.eos_idx,
            self.pad: self.pad_idx,
            self.unk: self.unk_idx,
        }
        self._idx2token = {idx:token for token, idx in self._token2idx.items()}
        
        
        idx = len(self._token2idx)
        min_freq = 0 if min_freq is None else min_freq
        max_freq = len(counter) if max_freq is None else max_freq
        
        for token, count in counter.items():
            if count > min_freq and count < max_freq:
                self._token2idx[token] = idx
                self._idx2token[idx]   = token
                idx += 1
        
        self.vocab_size = len(self._token2idx)
        self.tokens     = list(self._token2idx.keys())
    
    def token2idx(self, token):
        return self._token2idx.get(token, self.pad_idx)
    
    def idx2token(self, idx):
        return self._idx2token.get(idx, self.pad)
    
    def sent2idx(self, sent):
        return [self.token2idx(i) for i in sent]
    
    def idx2sent(self, idx):
        return [self.idx2token(i) for i in idx]
    
    def __len__(self):
        return len(self._token2idx)