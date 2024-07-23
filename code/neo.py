import sys 
import built_in
import readline


OPEN = '('
CLOSE = ')'

CIFRAS = "0123456789"


# TODO implementar el paso de un numero indefinido de variabless como argumento
# a un closure
# TODO implementar comentarios con ;
# TODO implementar que se pueda invocar eval desde el lenguaje
# TODO implementar que se pueda invocar apply desde el lenguaje



class env_EOF:
    def __init__(self):
        pass

    def eprint(self):
        print("EOF")

    def f_from(self,sym):
        return None

    def f_append(self,sym,f):
        pass

    def copy(self):
        return self

class ERROR:
    def __init__(self):
        pass

    def eprint(self):
        print("ERROR")

    def f_from(self,sym):
        return None

    def f_append(self,sym,f):
        pass

    def copy(self):
        return self


class Transformer:
    def __init__(self, args, pattern, template, env):
        self.args = args
        self.pattern = pattern
        self.template = template
        self.env = env

def match_pattern(pattern, args):
    asignations = {}
    if pattern == None:
        if args == None:
            return True
        else:
            return False
    if pattern.car == ".":
        return True
    if pattern.car == args.car:
        return match_pattern(pattern.cdr,args.cdr)
    return False


def apply_transformer(transformer, args):
    # does args match the pattern?
    # if not return None
    # if yes, apply the template
    # return the result
    
    assignations = {}
    aux_args = args
    aux_pattern = transformer.pattern
    while aux_pattern != None:
        if aux_pattern.car == ".":
            assignations[aux_pattern.cdr.car] =  aux_args
            aux_args = None
            aux_pattern = None
            break
        transformer.env.f_append(aux_pattern.car,aux_args.car)
        aux_pattern = aux_pattern.cdr
        aux_args = aux_args.cdr

DEBUG = True


def merror(msg):
    # print(msg,file=sys.stderr)
    raise Exception(msg)
    # exit(1)


class SchemeString:
    def __init__(self,data):
        self.data = data

class Closure:
    def __init__(self, params, body, env,is_built_in,name=None):
        self.name = name
        self.params = params
        self.body = body
        self.env = env
        self.is_built_in = is_built_in


    def __str__(self) -> str:
        c = stringify(self)
        return c


def eval_args(args,env):
    if DEBUG: print("evaluating arg: ",stringify(args))
    if args == None:
        return None
    kar = seval(args.car, env)
    return Pair(kar,eval_args(args.cdr,env))



def expand_env(closure,env):
    if closure.is_built_in:
        new_env = EnvironmentChain()
        new_env.append_chain(env)
        return new_env
    else:
        new_env = closure.env.copy()
        new_env.append_chain(env)
        return new_env
    


def mapply(closure,args,env):

    if DEBUG:
        print("EVALUATING CLOSURE: ",closure)
        print("NAME: ",closure.name)
        print("PARAMS: ",closure.params)
        print("BODY: ",closure.body)
        print("args: ",args)
        print("PARAMS: ", stringify(closure.params))
        print("ARGS: ", stringify(args))
        # env.eprint()	

    args = eval_args(args,env)

    if DEBUG: print("EVALUATED ARGS: ", stringify(args))

    new_env = expand_env(closure,env)
    if closure.is_built_in:
        return closure.body(args,env)
    
    else:
        if DEBUG : print("DEBUG: new_env")
        # new_env.eprint()
        aux_args = args
        aux_params = closure.params

        # comprobar que el numero de argumentos y parametros es el mismo
        lenargs = length(args)
        lenparams = length(closure.params)
        # if lenargs != lenparams:
        # 	merror("MAPPLY: Mismatched number of arguments and parameters"
        #   + "in closure " + closure.name + " with args "
        #   + stringify(args) + " " + stringify(closure.params))

        if DEBUG: print("DEBUG: aux_args = ",aux_args)
        if DEBUG: print("DEBUG: aux_params = ",aux_params)
        while aux_params is not None and aux_args is not None:
            # dot notation
            if aux_params.car == ".":
                new_env.f_append(aux_params.cdr.car,aux_args)
                aux_args = None
                aux_params = aux_params.cdr.cdr
                break

            new_env.f_append(aux_params.car, aux_args.car)
            aux_params = aux_params.cdr
            aux_args = aux_args.cdr


        if DEBUG: print("DEBUG: aux_args = ",aux_args)
        if DEBUG: print("DEBUG: aux_params = ",aux_params)

        if aux_params is not None or aux_args is not None:
            merror("Mismatched number of arguments and parameters MAPPLY")

        return seval(closure.body,new_env)


class Environment:
    def __init__(self,symbol,function,outer=None):
        self.symbol = symbol
        self.function = function
        self.outer = outer

    def eprint(self):
        print("{} : {}".format(self.symbol,self.function))
        if self.outer != None:
            self.outer.eprint()


    def append(self, sym,f):
        if self.symbol == sym:
            self.function = f
        if self.outer == None:
            self.outer = Environment(sym,f,None)
        else:
            self.outer.append(sym,f)
    
    def function_from(self,sym):
        if self.symbol == sym:
            return self.function
        else:
            if self.outer == None:
                return env_EOF()
            else:
                return self.outer.function_from(sym)
            
    def copy(self):
        if self.outer == None:
            return Environment(self.symbol,self.function,None)
        else:
            return Environment(self.symbol,self.function,self.outer.copy())




class EnvironmentChain:
    def __init__(self,env=None,outer=None):
        self.env = env
        self.outer = outer


    def append(self, env : Environment):
        if self.outer == None:
            self.outer = EnvironmentChain(env,None)

        else:
            self.outer.append(env)

    def append_chain(self,chain):
        if self.outer == None:
            self.outer = chain
        else:
            self.outer.append_chain(chain)

    def pop(self):
        if (self.outer == None):
            merror("Void environment cant pop")
        env = self.outer;
        self.outer = None;
        return env

    def f_append(self,sym,f):
        if self.env == None:
            self.env = Environment(sym,f)
        else:
            self.env.append(sym,f)

    def f_from(self,symbol):
        f = self.env.function_from(symbol)
        if f == env_EOF:
            if self.outer == None:
                return None
            else:
                return self.outer.f_from(symbol)
        else:
            return f

        # def f_append(sym,f):
        # 	self.env.append(sym,f)

    def eprint(self):
        print("Envirnment -----------")
        self.env.eprint()
        if self.outer != None:
            self.outer.eprint()


    def copy(self):
        if self.outer == None:
            return EnvironmentChain(self.env.copy(),None)
        else:
            return EnvironmentChain(self.env.copy(),self.outer.copy())



        


class Pair:
    def __init__(self,car,cdr):
        self.car = car
        self.cdr = cdr

    def __str__(self):
        return "( {} . {} )".format(str(self.car), str(self.cdr))
    
    def __format__(self,fmt):
        return self.__str__()

    def add_tail(self,item):
        if self.cdr == None:
            self.cdr = Pair(item,None)
        else:
            self.cdr.add_tail(item)
    
    def mprint(self):
        print("( ",end=" <__main__.Closure object at 0x7d7564081c90>")
        if type(self.car) == Pair:
            self.car.mprint()
        else:
            print(self.car,end=" ")
        if self.cdr == None:
            print(")",end=" ")	
        else:
            self.cdr.rprint()

    def rprint(self):
        if type(self.car) == Pair:
            self.car.mprint()
        else:
            print(self.car,end=" ")
        if self.cdr == None:
            print(")",end=" ")	
        else:
            self.cdr.rprint()



def copy_pair(p: Pair):
    if type(p) != Pair:
        return p
    else:
        return Pair(copy_pair(p.car),copy_pair(p.cdr))
    

def length(p:Pair):
    if p == None:
        return 0;
    return 1+length(p.cdr)


"""
aucixiliares
"""
"""
determina si el numero de elementos del par que se pasa como
argumento e le correcto
"""
def assert_arg_num(name: str, p: Pair,n: int):
    actual = length(p)
    if actual != n:
        merror("{} expected {} args but get {}".format(name,n,actual))
    
"""
Built in
"""

"""
true si el numeor es un netero (123) o un float (23.2312) con punto
false si no
"""
def is_number(c : str):
    has_coma = False
    for i in c:
        if i == '.':
            if has_coma:
                return False
            else:
                has_coma = True
        if i not in CIFRAS:
            return False
    return True
        

def is_bool(c):
    if c in ["#f","#t"]:
        return True
    return False
                

def is_blank(s):
    for c in s:
        if c not in " \t\n":
            return False
    return True

def tokenize(code : str):
    code = code.replace("'()","null")
    code = code.replace('[',' ( ').replace(']',' ) ')
    code = code.replace('(',' ( ').replace(')',' ) ')
    code = code.replace('\n',' ')
    code = code.replace('\t',' ')
    code = code.replace("'"," ' ")
    code = code.replace(","," , ")
    code = code.replace("`"," ` ")
    tokens = code.split(' ')
    tokens = [t for t in tokens if t != '']
    tokens = [t for t in tokens if not is_blank(t)]

    # expand the quotes
    
    # print("PRE tokens = ",tokens)
    # i = 0
    # for i in range(0,len(tokens)):
    # 	if t[0] == "'":
    # 		d = t[1:]
    # 		print("d = ",d)
    # 		print("aux = ",aux)
    # 		print("tokens = ",tokens)
    # 		tokens = aux + ['(','quote',d,')'] + tokens[i+1:]
    
    if DEBUG: print("DEBUG: tokenize return: ",tokens)
    return tokens


def pre_procesado(code, env):
    if DEBUG: print("pre_procesado")
    if DEBUG: print("code = ",code)
    if DEBUG: print("pcode = ",stringify(code))

    old_code = copy_pair(code)
    if code == None:
        return None
    if type(code) == Pair:
        if code.car == "'":
            kar = Pair("quote",None)
            kdr = code.cdr.car.cdr
            if DEBUG: print("EXPANDING quote kar = ",kar)
            if DEBUG: print("EXPANDING quote kdr = ",kdr)
        if code.car == "`":
            kar = Pair("quasiquote",None)
            kdr = code.cdr.car.cdr
            if DEBUG: print("EXPADING quasiquote kar = ",kar)
            if DEBUG: print("EXPADING quasiquote kdr = ",kdr)
        if code.car == ",":
            kar = Pair("unquote",None)
            kdr = code.cdr.car.cdr
            if DEBUG: print("EXPADING unquote kar = ",kar)
            if DEBUG: print("EXPADING unquote kdr = ",kdr)


    return old_code


        


def parse(tokens: list):
    if not tokens:
        return None

    token = tokens.pop(0)
    if token not in [OPEN, CLOSE,"'","`",","]:
        return token

    ## TODO poner ifs para expandir las quotes y los ( . ) 

    # expanssion de quote etc
    print ("token = ",token)
    if token == "'":
        new_pair = Pair("quote",None)
        new_pair.add_tail(parse(tokens))
        return new_pair
    
    if token == "`":
        new_pair = Pair("quasiquote",None)
        new_pair.add_tail(parse(tokens))
        return new_pair
    
    if token == ",":
        new_pair = Pair("unquote",None)
        new_pair.add_tail(parse(tokens))
        return new_pair
    
    if token == OPEN:
        new_pair = Pair(None, None)
        new_pair.car = parse(tokens)
        while tokens:
            if tokens and tokens[0] == CLOSE:
                tokens.pop(0)
                break
            new_pair.add_tail(parse(tokens))
            
        return new_pair

    return None


def is_number(s:str):
    C = "1234567890"
    has_point = False
    for i in s:
        if i not in C+'.':
            return False
        if i == '.':
            if has_point:
                return False
            has_point = True

    return True

def is_int(s:str):
    try:
        int(s)
        return True
    except:
        return False


def eval_atom(atom, env):
    # is string 

    if atom == "null":
        return None
    
    if atom == "#t":
        return True
    
    if atom == "#f":
        return False
    
    if type(atom) in [bool, int, float, SchemeString]:
        return atom

    if atom[0] == '"' and atom[-1] == '"':
        return SchemeString(str(atom[1:-1]))
        return str(atom[1:-1])
        
    # is int?
    try:
        d = int(atom)
        return d
    except:
        pass

    # is float?
    try:
        d = float(atom)
        return d
    except:
        pass

    # if is nothing ele is a symbol
    # print("buscando {} en el env".format(atom))

    # env.eprint()

    if atom == "x":
        pass

    d = env.f_from(atom)
    if d != env_EOF:
        # print("encontrado: ",d)
        return d
    else:
        merror("Symbol {} not in enviroment".format(atom))
    # try:
    # 	assert_identifier(atom)
    # except:
    # 	pass

    # return atom


""" # REDUNDANTE
def eval_symbol( args, env):
    symbol = args.car
    f = env.f_from(symbol)
    if f == None:
        merror("Symbol {} not in enviroment".format(symbol))
    return f
"""	

# IMPLEMENTAR ATOM? y LIST? o PAIR?



    
def is_special_form(string):
    if string in ["define","lambda","quote","car","cons",
               "cond","null","menv","eval","begin",
               "quasiquote","unquote","quote"]:
        return True
    return False


def is_symbol(sym,env):
    if env.f_from(sym) != None:
        return True
    return False
    

#definir un stringify para mostrar las cosas en formato lisp 




def assert_atom(code, name=""):
    if type(code) == Pair:
        merror("Expected atom but encounter list: " + str(code) + name)

def assert_list(code, name=""):
    if type(code) != Pair:
        merror("Expected list but encounter atom: " + str(code)+" method: "+ name)

def assert_num_args(n,args, name=""):
    assert_list(args)
    if length(args) != n:
        merror("error: "+name+" expected {} arguments but {} were given : {}".format(n,length(args),args))

def quote(args,env):
    assert_num_args(1,args,"quote")
    return args.car

def unquote(args,env):
    assert_num_args(1,args,"unquote")
    return args.car

def unquote_from_quasiquote(args,env):
    
    kar = args.car
    if kar == None:
        return None
    if kar.car == "unquote":
        return seval(kar.cdr.car,env)
    else:
        return Pair(seval(kar,env),unquote_from_quasiquote(args.cdr,env))

def quasiquote(args,env):
    assert_num_args(1,args,"quasiquote")
    kar = unquote_from_quasiquote(args,env)
    return kar

    

def iatom(args):
#	assert_num_args(1,args,"atom")
    if type(args) == Pair:
        return False
    return True


def atom(args,env):
    assert_num_args(1,args,"atom")
    if type(args.car) == Pair:
        return False
    return True

def eq(args,env):
    assert_num_args(2,args,"eq")
    # print("EQ: code = ",stringify(args), " type = ",type(args))
    # print("EQ: args.car = ",stringify(args.car), " type = ",type(args.car))
    # # print("EQ: args.cdr = ",stringify(args.cdr), " type = ",type(args.cdr))
    # print("EQ: args.cdr.car = ",stringify(args.cdr.car), " type = ",type(args.cdr))

    A = args.car
    B = args.cdr.car
    # assert_atom(A)
    # assert_atom(B)
    if type(A) != type(B):
        return False
    if A == B:
        return True
    return False

def mar(args):
    assert_list(args,"car")
    return args.car

def car(args,env):
    if DEBUG: print("CAR: code = ",stringify(args), " type = ",type(args),"car = ",args.car)

    assert_num_args(1,args,"car")
    assert_list(args,"car")

    if DEBUG: print("CAR: args.car = ",args.car)

    args = args.car
    args = seval(args,env)
    kar = args.car

    if DEBUG: print("CAR: kar = ",kar)

    return kar
    

def cdr(args,env):
    assert_list(args,"cdr")
    if DEBUG: print(stringify(args))
    if DEBUG: 
        print("CDR: code = ",stringify(args), " type = ",type(args))
        print("CDR: args.car = ",stringify(args.car), " type = ",type(args.car))
        print("CDR: args.car.cdr = ",stringify(args.car.cdr), " type = ",type(args.car.cdr))
    return args.car.cdr

def cadr(args):
    assert_list(args,"cadr")
    kdr = args.cdr
    assert_list(kdr)
    return mar(kdr)
    

def cons(args,env):

    # print("CONS: code = ",stringify(args), " type = ",type(args))
    # print("CONS: car = ",stringify(args.car), " type = ",type(args.car))
    # print("CONS: cdr = ",stringify(args.cdr), " type = ",type(args.cdr))
    # print("CONS: cadr = ",stringify(args.cdr.car), " type = ",type(args.cdr.car))
    assert_num_args(2,args,"cons")
    kar = seval(args.car,env)
    kdr = seval(args.cdr.car, env)
    # print ("CONS: kar = ",kar)
    # print ("CONS: kdr = ",kdr)

    return Pair(car = kar, cdr = kdr)

def cond(args,env):
    if args == None:
        return None
    # print("COND: code = ",stringify(args), " type = ",type(args))
    # print("COND: args.car = ",stringify(args.car), " type = ",type(args.car))

    primer = args.car
    condicion = seval(primer.car,env)
    cuerpo = primer.cdr.car
    if condicion:
        return seval(cuerpo,env)
    else:
        return cond(args.cdr,env)


def begin(args,env):
    if args == None:
        return None
    if args.cdr == None:
        return seval(args.car,env)
    else:
        seval(args.car,env)
        return begin(args.cdr,env)
        

def mlambda(args, env):
    params = mar(args)
    body = mar(args.cdr)

    return Closure(params,body,env,False)
    


def eval_special_form(form,args,env):
    if form == "atom":
        return iatom(args)
    if form == "eq":
        return eq(args)
    if form == "car":
        return car(args,env)
    if form == "define":
        return eval_define(args,env)
    if form == "cons":
        return cons(args,env)
    if form == "lambda":
        return mlambda(args, env)
    if form == "cond":
        return cond(args,env)
    if form == "quote":
        return quote(args,env)
    if form == "eval":
        return seval(args.car,env)
    if form == "begin":
        return begin(args,env)
    if form == "quasiquote":
        return quasiquote(args,env)

    
    if form == "menv":
        env.eprint()
        return None



def apply_built_in(sym, args, env):
    f = env.f_from(sym)
    kar = seval(mar(args),env)
    kdr = seval(mar(args.cdr),env)
    r = f(kar, kdr)
    return r



def assert_identifier(s):
    cannotbegin = ".+-1234567890"
    if s == None:
        merror(f"Syntax error: {s} no es un identificador")
    if s[0] in cannotbegin:
        merror(f"Syntax error: {s} no es un identificador")
    permited_chars = "!$%&*/:<=>?~_^"
    num = "1234567890"
    alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    all = permited_chars+num+alpha
    for i in s[1:]:
        if i not in all:
            merror(f"Syntax error: {s} no es un identificador")


# TODO toquetear define porque una 
# variable no es autoevaluable si no tiene 
# un valor asociado en el env 
def eval_define(args, env):

    if DEBUG: print("DEFINE: code = ",stringify(args), " type = ",type(args))	
    assert_num_args(2,args,"DEFINE")
    kar = mar(args)
    if type(kar) == Pair:
        kar = seval(kar,env)

    assert_identifier(kar)

    # kar = seval(mar(args),env)

    kdr = seval(mar(args.cdr),env)

    if type(kdr) == Closure:
        kdr.name = kar
    if callable(kdr):
        kdr.name = kdr.__name__
    
    # if DEBUG: print("DEBUG: definiendo ",kar," = ",kdr)
    # if DEBUG: print("DEBUG: tipos ",type(kar),type(kdr))

    assert_identifier(kar)
    env.f_append(kar,kdr)
    if DEBUG: print(f"DEBUG: definido {kar} = {kdr}")


def is_exiting_procedure(sym,env):
    if is_special_form(sym):
        return True
    d = env.f_from(sym)
    if d == env_EOF:
        return False
    if type(d) == Closure or callable(d):
        return True
    else:
        return False

def assert_existing_procedure(sym,env):
    if not is_exiting_procedure(sym,env):
        merror(f"ERROR: Expexted a Procedure: {sym}")

def seval(code,env):
    if DEBUG: print("evaluating: code", stringify(code))

    if iatom(code):
        return eval_atom(code, env)
    else:
        kar = mar(code)
        # TODO quitar essto es debug
        if kar == "lambda":
            print("aaaa: ",stringify(code))
            pass
        args = code.cdr

        if is_special_form(kar):
            return eval_special_form(kar,args,env)

        kar = seval(kar,env)

        # TODO asertear que el car es un procedure
        # si no es un procedure  error :)
        # assert_existing_procedure(kar,env)


        if type(kar) == Closure:
            return mapply(kar, args,env)
        
        # error
        merror("ERROR: Expected a procedure but encounter: " + str(kar))

        # if is_special_form(kar):
        # 	return eval_special_form(kar,args,env)
        # kdr = code.cdr




## REDO esto
            


# an auxiliar function to print the tree in lissp form 
def stringify(code):
    c = ""
    if type(code) == Pair:
        c += "( "
        c += stringify(code.car)
        # print all the args
        aux = code.cdr
        while aux != None:
            if type(aux) == Pair:
                c += " " + stringify(aux.car)
            else:
                c += " . " + stringify(aux)
                break
            aux = aux.cdr
        c += ") "

    elif type(code) == Closure:
        c += "Procedure: <"
        c += stringify(code.params) + " "
        c += stringify(code.body) + " "
        c += ">"
    elif type(code) == bool:
        if code:
            c = "#t "
        else:
            c = "#f "
    elif code == None:
        c = "null "
    else:
        c = str(code)+" "

    return c



# test stringify
code = '( + 4 \'eval)'
code = '(+ (- 10 2) 55 (quote asd) (length "ad"))'
print(code)
tokens = tokenize(code)
print(tokens)
arbol = parse(tokens)
print(arbol)
stringify(arbol)

    

def suma(args, env):
    if args == None:
        return 0
    kar = seval(mar(args),env)
    # print("kar = ",kar)
    # print("type = ",type(kar))
    return kar + suma(args.cdr,env)

def resta(args, env):
    if args == None:
        return 0
    kar = seval(mar(args),env)
    return kar - resta(args.cdr,env)

def multiplicacion(args, env):
    if args == None:
        return 1
    kar = seval(mar(args),env)
    return kar * multiplicacion(args.cdr,env)

# TODO la division funciona regular
def division(args, env):
    if args == None:
        return 1
    kar = seval(mar(args),env)
    return kar / division(args.cdr,env)




def eval_string(s,menv):
    try:
        t = tokenize(s)
        if DEBUG: print("DEBUG: EVAL_STRING tokens = ",t)
    except Exception as e:
        print("Tokenization ERROR: ",e)
        return ERROR()
    return block_evaluation(t)


def i_read_file(file):
    f = open(file,"r")
    code = ""
    l = f.readline()
    while l != "":
        # read only until ";"
        aux = ""
        c = l[0]
        i = 1
        while c != ";" and i < len(l):
            aux += c
            c = l[i]
            i += 1

        code += aux
        l = f.readline()

    return code



def block_evaluation(tokens):
    aux = tokens.copy()
    try:
        p = parse(aux)
        if DEBUG: print("DEBUG: block_evaluation: code = ",p)
        auxp = copy_pair(p)
        try:
            r = seval(p,menv)
            print(stringify(r))
            return r
        except Exception as e:
            print("ERROR: ",e)
            print("ERROR: ",stringify(auxp))
            return ERROR()
    except Exception as e:
        print("Syntax Error: ",e)
        print("Syntax Error: "," ".join(aux))
        return ERROR()


def eval_file(file,menv):
    code = i_read_file(file)

    t = tokenize(code)
    if DEBUG: print("DEBUG: EVAL_FILE tokens = ",t)

    block = []
    parentesis_count = 0
    while t:
        token = t.pop(0)
        if token == "(":
            parentesis_count += 1
        if token == ")":
            parentesis_count -= 1
        block.append(token)
        if parentesis_count == 0:
            block_evaluation(block)
            block = []
            parentesis_count = 0
            continue

    


def internal_test(env):

    print("TEST null")
    print(menv.f_from('null'))

    print("TEST, lambda")
    eval_string("null",menv)
    c = "(lambda (x) (+ x x))"
    d = seval(parse(tokenize(c)),menv)
    print(d)
    stringify(d)

    print("TEST, define lambda")
    eval_string("(define a (lambda (v) (+ v 1)))",menv)
    eval_string("(a 2)",menv)

    print("TEST cons")
    eval_string("(cons 1 2)",menv)
    eval_string("(cons 1 null)",menv)

    print("TEST car")
    eval_string("(car (cons 1 2))",menv)
    eval_string("(car (cons 1 (cons 2 null)))",menv)
    eval_string("(car (cons 1 (cons 2 (cons 3 null))))",menv)

    print("TEST car sobre lista definida")
    eval_string("(define lista (cons 1 (cons 2 (cons 3 null))))",menv)
    eval_string("lista",menv)
    eval_string("(car lista)",menv)

    print("TEST cdr")
    eval_string(" (define lista (cons 1 (cons 2 (cons 3 null))))",menv)
    eval_string("lista",menv)
    eval_string("(cdr lista)",menv)

    print("TEST atom: t f t f t")
    eval_string("(atom 1)",menv)
    eval_string("(atom (cons 1 2))",menv)
    eval_string("(atom null)",menv)
    eval_string("(atom (cons 1 null))",menv)
    eval_string("(atom (lambda (x) (+ x x)))",menv)

    print("TEST eq? true")
    eval_string("(eq? 1 1)",menv)
    eval_string("(eq? 2 2)",menv)
    eval_string("(eq? 1.0 1.0)",menv)
    eval_string("(eq? (+ 3 4) 7)",menv)
    eval_string("(eq? + +)",menv)
    eval_string("(eq? eq? eq?)",menv)

    print("TEST eq? false")
    eval_string("(eq? 1 2)",menv)
    eval_string("(eq? 1 1.0)",menv)
    eval_string("(eq? (cons 1 2) (cons 1 2))",menv)

    print("TEST cond")
    eval_string("(define n 2)",menv)
    eval_string("""
        (cond 
            ((eq? n 1) 1)
            ((eq? n 2) 2)
            ((eq? n 3) 3)
        )
    """,menv)
    eval_string("""
        (cond 
            (#t 1)
            (#f 2))
    """,menv)
    eval_string("""
        (cond)
    """,menv)
    
    print("TEST quote")
    eval_string("(quote 1)",menv)
    eval_string("(quote (1 2 3))",menv)


    print("TEST, other")





if __name__ == "__main__" :
    
    code = '( + 4 \'eval)'
    code = '(+ 3 4)'
    print(code)
    tokens = tokenize(code)
    print(tokens)
    arbol = parse(tokens)
    print(arbol)
    arbol.mprint()
    print("\n\ntamaño del arbol = ", length(arbol))


    # OUTDATED
    # # menv = EnvironmentChain(Environment('+',lambda x: x.car + x.cdr.car))
    # menv = EnvironmentChain(Environment('+',lambda x,y: x + y))

    # # menv.f_append('-', lambda x: x.car - x.cdr.car)
    # menv.f_append('-', lambda x,y:x - y)
    # menv.f_append('*', lambda x,y:x * y)
    # menv.f_append('/', lambda x,y:x / y)



    # print(menv.f_from('+'))
    # print(menv.f_from('-'))
    # print(menv.f_from('*'))
    # print(menv.f_from('/'))



    menv = EnvironmentChain()
    menv.f_append('+', Closure(None, suma, None, True))
    menv.f_append('-', Closure(None, resta, None, True))
    menv.f_append('*', Closure(None, multiplicacion, None, True))
    menv.f_append('/', Closure(None, division, None, True))

    menv.f_append('cdr', Closure(None, cdr, None, True))
    menv.f_append('atom', Closure(None, atom, None, True))
    menv.f_append('eq?', Closure(None, eq, None, True))

    menv.f_append('null', None)

    menv.eprint()

    # seval(arbol, menv)

    eval_string("""
    (define null? (lambda (x)
                    (eq? x null)))
    """,menv)

    eval_string("""
    (define sumalista (lambda (lista)
                        (cond ((null? lista) 0)
                            (#t (+ (car lista) (sumalista (cdr lista)))))))
    """, menv)

    eval_string("""
    (define summ (lambda (a . args)
                (+ a (sumalista args))))
    """, menv)
    eval_string("""
    (summ 1 2 3 4 5 6 7 8 9 10)
    """, menv)


    
    print("sys.argv = ",sys.argv)

# 	eval_string("""
# 			 (define null? (lambda (x)
# 				 (eq? x null)))
# 			 """,menv)
    

# 	eval_string("""
# (define sumalista (lambda (x)
#   (cond ((null? x) 0)
#         (#t (+ (car x) (sumalista (cdr x)))))))

# 	""",menv)
# 	eval_string("""
# (define m_suma (lambda (a . l)
#                  (+ a (sumalista l))))

# 			 """,menv)


# 	eval_string("""
             
# (m_suma 1 2 )
# 	""",menv)

    eval_string("(lambda () 3)",menv)

    # eval_string("""
    # 	(define sumalista (lambda (x)
    # 		(cond 	((atom x) x)
    # 				(#t (cons 	(+ (car x) 1)
    # 							(sumalista (cdr x))))
    # 		)
    # 	))
    # """,menv)
    # eval_string("(sumalista '(1 2 3 4))",menv)

    # eval_file("a.scm",menv)

    if len(sys.argv) > 1:
        eval_file(sys.argv[1],menv)
    else:
        # a repl
        while True:
            # use readline 
            code = input("neo > ")
            if code == "exit":
                break
            if code != "":
                eval_string(code,menv)
