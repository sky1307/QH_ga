import random
import yaml
from ga.Population import Population


class GA:
    with open("settings/ga/setting.yaml", 'r') as stream:
        config =yaml.load(stream ,Loader=  yaml.FullLoader)
    SIZE_POPULATION = config['SIZE_POPULATION']
    CONDITION_STOP =  config['CONDITION_STOP'] 
    pc = config['pc']
    pm = config['pm']
    
    def __init__(self , sigma ,  fitness):
        f0 = open("log/ga/init.txt", 'a+')
        f0.write("Hello")
        f0.close()
        print("hello>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        self.pop = Population(size = GA.SIZE_POPULATION, sigma = sigma,f = fitness)
     
        print("--------------------close----------------------------")
        print("-")
        print("-")
        print("-")
        print("-")
        print("-")
        print("-")
        print("-")
        print("-")
        print("-")
        

    def crossover_mutation(self):
        a = random.randint(0, GA.SIZE_POPULATION -1 )
        b = random.randint(0, GA.SIZE_POPULATION -1 )
        while a == b:
            b = random.randint(0, GA.SIZE_POPULATION -1)
        print(a," ",b)
        ind1 = self.pop.pop[a]
        ind2 = self.pop.pop[b]
        p = random.random()
        if p < GA.pc:
            return self.pop.crossover(ind1,ind2)
        elif p < GA.pc + GA.pm:
            return self.pop.mutation(ind1) + self.pop.mutation(ind2)
        else:
            return self.crossover_mutation()
    def run(self):
        
        i = self.pop.k
        while i < GA.CONDITION_STOP:
            file_name="log/ga/population"+str(i+1)+".txt"
            child = []
            while len(child) < GA.SIZE_POPULATION:
                child += self.crossover_mutation()
            self.pop.pop += child
            self.pop.selection()
            self.pop.get_best()
    


            fi = open(file_name,'w+')
            fi.write(str(i+1))
            fi.write('\n')
            fi.close()
            for x in self.pop.pop:
                x.write_file(file_name,'a+')

            f2 = open("log/ga/runtime.txt","a+")
            f2.seek(0,2)
            print("\n+++++++++++++++Chon loc lan thu : ",i+1,"+++++++++++++++++++\n")
            f2.write("\n+++++++++++++++Chon loc lan thu : "+str(i+1)+"+++++++++++++++++++\n")
            print("+")
            print("+")
            print("+")
            f1 = open("log/ga/run.txt", 'a+')
            f1.seek(0,2)
            f1.write("\n----------------the he: "+str(i+1)+"--------------\n")
            for j in range(GA.SIZE_POPULATION):
                f1.write(self.pop.pop[j].__str__())
                f1.write("\n")
            
            f1.close()
            f2.write("\n+++++++++++++++Chon  loc  xong : "+str(i+1)+"+++++++++++++++++++\n")
            print("-----------------------------------------------------")
            f2.close()
            i +=1
        return 0    

