class Relation:
    """
    This class keeps track of each edge present between the source and the target and contains all the information 
    necessary to recognize the relation
    Attributes:
        type_relation: (string) type of edge (c2a, c2q, a2q)
        source: (classe User) indicate node source
        target: (classe User) indicate node target
        weight: (int) weight of edge
        time: (int) must have passed like this YYYYMM, and indicates the moment in which the relation took place 
    """
    def __init__(self, type_relation, time, source, target, weight):
        self.type_relation_ = type_relation
        self.time_ = time
        self.source_ = source.get_ID
        self.target_ = target.get_ID
        self.weight_ = weight

    @property
    def get_type(self):
        return self.type_relation_
    
    @property
    def time(self):
        return self.time_
    
    @property
    def target(self):
        return self.target_
    
    @property
    def source(self):
        return self.source_
    
    def set_weight(self, weight):
        self.weight_ = weight
    
    @property
    def weight(self):
        return self.weight_
    
    def __str__(self): 
        return "{\"type_relation\": \"" + self.type_relation_ + "\", \"time\": " + str(self.time_) + ", \"source\": " + str(self.source_) + ", \"target\": " + \
        str(self.target_) + ", \"weight\": "+ str(self.weight_) + "}"
    
    def __repr__(self): 
        return self.__str__()
     

class User:
    """
    this class contains information about the user and his relations
    Attributes:
        ID_user: (int) ID of user
    """
    def __init__(self, ID_user):
        self.ID_user = ID_user
        self.in_relation = dict()
        self.out_relation = dict()
    
    def add_in_relation(self, in_relation):
        """
        Add relation in User with incoming edge
        
            Parameters:
                    in_relation (Relation) relation between self and v
        """
        # we groupby all relation of user that have the same time so if the time not exists we create it
        # and under each relation we have all type edge (c2a, c2q, a2q)
        if in_relation.time in self.in_relation:
            if in_relation.get_type not in self.in_relation[in_relation.time]:
                self.in_relation[in_relation.time][in_relation.get_type] = []
        else:
            self.in_relation[in_relation.time] = {in_relation.get_type: []}
        self.in_relation[in_relation.time][in_relation.get_type].append(in_relation)

    
    def add_out_relation(self, out_relation):
        """
        Add relation in User with outgoing edge
        
            Parameters:
                    in_relation (Relation) relation between self and u
        """
        # we groupby all relation of user that have the same time so if the time not exists we create it
        # and under each relation we have all type edge (c2a, c2q, a2q)
        if out_relation.time in self.out_relation:
            if out_relation.get_type not in self.out_relation[out_relation.time]:
                self.out_relation[out_relation.time][out_relation.get_type] = []
        else:
            self.out_relation[out_relation.time] = {out_relation.get_type: []}
        self.out_relation[out_relation.time][out_relation.get_type].append(out_relation)
    
    def set_in_relation(self, inRelations):
        self.in_relation = inRelations
    
    def set_out_relation(self, outRelation):
        self.out_relation = outRelation
    
    @property
    def get_ID(self):
        return self.ID_user

    @property
    def get_in_relation(self):
        return self.in_relation
    
    @property
    def get_out_relation(self):
        return self.out_relation

    def __str__(self):
        return "{\"in_relation\": " + str(self.in_relation) +  ", \"out_relation\": " + str(self.out_relation) + "}"

    def __repr__(self): 
        return self.__str__()