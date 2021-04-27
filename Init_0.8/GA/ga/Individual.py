import random
import yaml
import datetime
import pytz


class Individual:
    with open("settings/ga/setting.yaml", 'r') as stream:
        config =yaml.load(stream ,Loader= yaml.FullLoader)
    n = config['n']
    pi= config['pi']
    psm = config['psm']
    nmax = config['nmax']
    sm = config['sigma']
    def __init__(self, sigma):
        self.size = len(sigma)
        self.genes = [0] * self.size
        # check0 = 0
        if random.random() < Individual.pi:
            for i in range(self.size):
                if random.random() < 0.5:
                    self.genes[i] = 1
                # check0 += self.genes[i]
        else:
            for i in range(self.size):
                if random.random() < sigma[i] :
                    self.genes[i] = 1
                # check0 += self.genes[i]
        self.set_n(int(random.gauss(Individual.n, Individual.sm)))
        self.value_fitness = 0
        self.time = datetime.datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
        # self.value_fitness = 100000000

    def __str__(self):
        time = str(self.time)
        s = time +' --> : '+ "ssa ="
        for i in self.genes:
            s +=  str(i) + ' '
        s +="  n =" + str(self.n) + "  fitness = " + str(self.value_fitness)
        return s
    def write_file(self,file_name,t):
        s=''
        for i in self.genes:
            s +=  str(i) + ' '
        s += str(self.n) + ' '+ str(self.value_fitness)+' '+str(datetime.datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')))
        f = open(file_name,t)
        f.seek(0,2)
        f.write(s)
        f.write('\n')
        f.close()

    def get_genes(self):
        return self.genes
    def get_n(self):
        return self.n
    
    def set_genes(self, genes):
        for i in range(self.size):
            self.genes[i] = genes[i]
    def set_n(self, n):
        self.n = n
        if self.n<2:
            self.n = 2
        if self.n > Individual.nmax:
            self.n = Individual.nmax