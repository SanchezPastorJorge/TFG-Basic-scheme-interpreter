import readline
import sys

"""
	AUTOR: JORGE SANCHEZ PASTOR
"""

# ESTUDIAR PROPRER TAIL ECURSION PORQUE IMPORTA y OPTMIZACION etc
# Hygienic macro


class Closure:
    def __init__(self, params, body, env=[{}], built_in = False, name = ""):
        self.params = params
        self.body = body
        self.env = env
        self.built_in = built_in
        self.name = name

class Macro:
    def __init__(self, params, body, name=""):
        self.params = params
        self.body = body
        self.name = name


class Transformer:
    """
	# TODO implementar los macros de forma correcta
    El enviroment del transsformer es interno y no se ua
    para ampliar el enviroment mayor
    """
    def __init__(self, pattern, template, env):
        self.pattern = pattern
        self.template = template


class Error:
    def __init__(self):
        pass

    def __str__(self):
        return "Internal Error"

class EOF:
    def __init__(self):
        pass

class LispString:
    def __init__(self, string):
        self.string = string

    def __str__(self):
        return "\""+self.string+"\""
    
    def copy(self):
        return LispString(self.string)


DEBUG = False
APPLY_DEBUG = False
LAMBDA_DEBUG = False
DEBUG_EVAL_ATOM = False
OPEN = '('
CLOSE = ')'


#
# ----    ENVIROMENT    ----
#
# un enviroment es un diccionaro el envorment que se usa es
# una lista de diccionarios

def is_in_environtment(symbol, env):
    for e in env:
        if symbol in e:
            return True
    return False

def get_value(symbol, env):
    for e in env:
        if symbol in e:
            return e[symbol]
    return None


#
# ----    PAIR MANIPULATION    ----
#

def copy_pair(pair):
    if pair == (): 
        return ()
    if atom_(pair): 
        return pair
    return (car(pair), copy_pair(cdr(pair)))


def pair_to_list(pair):
    if pair == ():
        return []
    return [car(pair)] + pair_to_list((cdr(pair)))


def list_to_pair(l):
    if l == []:
        return ()
    return (l[0], list_to_pair(l[1:]))


    


#
# ----    built-in functions    ----
#

def car(pair):
    return pair[0]

def cdr(pair):
    return pair[1]

def cadr(pair):
    return pair[1][0]

def cons(args):
    return (car(args), cadr(args))


def suma(par):
    return par[0] + par[1][0]

def resta(par):
    return par[0] - par[1][0]

def mult(par):
    return par[0] * par[1][0]

def div(par):
    return par[0] / par[1][0]


def null_(pair):
    return pair == ()

def mayor_(par):
    return par[0] > par[1][0]

def menor_(par):
    return par[0] < par[1][0]


def append(pair, value):
     if pair == ():
         return (value, ())
     return (car(pair), append(cdr(pair), value))

def length(pair):
    c = 1
    if pair == ():
        return 0;

    aux = cdr(pair)
    while aux != ():
        aux = cdr(aux)
        c += 1
    return c
        
def pair_(pair):
    if pair == (): return False
    return type(pair) == tuple

def atom_(pair):
    return not pair_(pair)

def eq_(pair):
    if length(pair) != 2:
        merror("eq? wrong number of arguments")
    if type(pair[0]) != type(pair[1][0]):
        return False
    return pair[0] == pair[1][0]



def identifier_(exp):
    initials = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    initials += "!$%&*/:<=>?^_~"
    digits = "0123456789"
    subseq = initials + digits + "+-.@"
    if exp == ():
        return False
    if pair_(exp):
        return False
    if exp[0] not in initials:
        return False
    for c in exp:
        if c not in subseq:
            return False
    return True


def procedure_(exp):
    return type(exp) == Closure

def peval(args, env):
    paso = car(args)
    return ieval(paso,  env)

def display(exp):
    print(stringify(car(exp)))
    return True

def newline(exp):
    print()
    return True

#
# ----    SPECIAL FORMS   ----
#


# !! LAMBDA NO EVALUA SU CUERPO EN LA DEFINICION
def ilambda(args, env):
    if length(args) != 2:
        merror("lambda: wrong number of arguments")
    if LAMBDA_DEBUG:
        print(" LAMBDA ")
        print("args = ",args)
        print("car(args) = ",car(args))
        print("cdr(args) = ",cdr(args))
        print("car(cdr(args)) = ",car(cdr(args))),

    return Closure(
        params = car(args),
        body = car(cdr(args)),
        env = env,
        built_in = False,   
    )

# define si evalua u cuerpo en la definicion
def define(args, env):
    if length(args) != 2:
        merror("define: wrong number of arguments")

    if type(car(args)) != str:
        # 2nd form of define , 
        kar = car(args)
        name = car(kar)
        parameters = cdr(kar)
        body = car(cdr(args))

        # c = ttp(('define', name, ('lambda', parameters, body)))
        c = ('define', (name, (('lambda', (parameters, (body, ()))), ())))
        print("nuevo define: ", c)
        ieval(c, env)
        return None

        

    env[0][car(args)] = ieval(cadr(args), env)

    return None

def defmacro(args, env):
    if length(args) != 3:
        merror("defmacro: wrong number of arguments")
    
    if type(car(args)) != str:
        merror("defmacro: first argument must be a symbol")


    name = car(args)
    args = cdr(args)
    params = car(args)
    args = cdr(args)
    body = car(args)


    macro = Macro(params, body, name)
    macros[name] = macro


def subst_values(body, internal_env):
    if body == ():
        return ()
    if atom_(body):
        if body in internal_env:
            return internal_env[body]
        return body
    return (subst_values(car(body), internal_env), subst_values(cdr(body), internal_env))

def expand_macro(code, env):
    if car(code) in macros:
        new_env = [{}] + env
        macro = macros[car(code)]

        params = macro.params
        new_code = macro.body
        

        # internal_env = {}
        lparams = pair_to_list(params)
        largs = pair_to_list(cdr(code))

        for i in range(0,len(lparams)):
            new_env[0][lparams[i]] = largs[i]
        new_code = ieval(new_code, new_env)
        return new_code


def expand_macros(code, env):
    if pair_(code):
        if car(code) in macros:
            return expand_macro(code, env)
        return (expand_macros(car(code), env), expand_macros(cdr(code), env))
    return code
        
        
    
    


def _cond(args, env):
    if args == ():
        return None
    
    estatement = car(args)
    q = car(estatement)
    body = cadr(estatement)
    rq = ieval(q, env)

    if DEBUG : print("DEBUG: cond: q = ",q)
    if DEBUG : print("DEBUG: cond: rq = ",rq)
    if DEBUG : print("DEBUG: cond: body = ",body)

    if rq :
        return ieval(body, env)
    return cond(cdr(args), env)



# def _cond(args, env):
#     if args == ():
#         return None
    
#     estatement = car(args)


#     if ieval(q, env) :
#         return ieval(cadr(estatement), env)
#     return cond(cdr(args), env)
"""
	Cond sin recusrion
"""
def cond(args, env):
    if args == ():
        return None
    estatements = pair_to_list(args)
    for e in estatements:
        q = ieval(car(e), env)
        if e == ():
            return None
        if q:
            return ieval(cadr(e), env)
    return None


def quote(exp):
    return car(exp)

def unquote_list(exp,env):
    if exp == ():
        return ()
    if type(exp) != tuple:
        return exp
    if car(exp) == "unquote":
        return ieval(cadr(exp),env)
    return (unquote_list(car(exp),env), unquote_list(cdr(exp),env))

def quasiquote(exp,env):
    exp = unquote_list(exp,env)
    return exp

def begin(exp,env):
    if exp == ():
        return None
    print("begin exp = ",exp)
    aux = exp
    while True:
        r = ieval(car(aux),env)
        aux = cdr(aux)
        if aux == ():
            return r
        

def syntax_rules(args,env):

    literals = car(args) # 

    matches = pair_to_list(cdr(args))
    

    match_1 = cadr(args)

    pattern = car(match_1)
    template = (cadr(match_1))
    if DEBUG: print("DEBUG, syntax-rule pattern", stringify(pattern))
    if DEBUG: print("DEBUG, syntax-rule template", stringify(template))

    if DEBUG: print("DEBUG, syntax-rule literals", stringify(literals))




    

#
# ----    EVALUATION    ----
#

SPECIAL_FORMS = ["define","cond","quote","lambda","quasiquote","begin","syntax-rules","defmacro"]

def eval_special_form(code, env):
    if car(code) == "define":
        return define(cdr(code), env)
    if car(code) == "quote":
        return quote(cdr(code))
    if car(code) == "lambda":
        return ilambda(cdr(code), env)
    if car(code) == "cond":
        return cond(cdr(code), env)
    if car(code) == "quasiquote":
        return quasiquote(car(cdr(code)),env)
    if car(code) == "begin":
        return begin(cdr(code),env)
    if car(code) == "syntax-rules":
        return  syntax_rules(cdr(code), env)
    if car(code) == "defmacro":
        return defmacro(cdr(code), env)
    


def eval_atom(atom, env):

    if DEBUG_EVAL_ATOM: print("EVAL_ATOM: atom =",atom)

    if atom == "null":
        return ()
    if atom == "#t":
        return True
    if atom == "#f":
        return False
    if atom == "else":
        return True
    
    # int 
    try: return int(atom)
    except: pass

    # float
    try: return float(atom)
    except: pass

    # string
    if atom[0] == '"' and atom[-1] == '"':
        return LispString(atom[1:-1])
    
    # symbol
    if is_in_environtment(atom, env):
        return get_value(atom, env)
    else:
        merror("undefined symbol: "+str(atom))
    


def ieval(code, env):

    #TODO remove
    if pair_(code):
        if DEBUG: print("IEVAL: going to evaluate: ",stringify(code))

        if car(code) == "+":
            pass
        if car(code) == "sumalista":
            pass

    if atom_(code):
        return eval_atom(code, env)
    
    if car(code) in SPECIAL_FORMS:
        return eval_special_form(code, env)
    
    kar = car(code)
    kdr = cdr(code)


    kar = ieval(car(code), env)

    if not procedure_(kar):
        merror("not a procedure: "+stringify(kar))

    


    r = iapply(kar, cdr(code), env)


    # atoms are evaluated when output for when they are from a quote
    # which means these atom are not being evaluated and are simple python
    # string without type. this cases trying to evaluate (), so we made and exception
    # TODO implementar eto mejor? evaluando todos loa atomos a u tipo correspondiente
    # antes de todo?
    if atom_(r) and r != ():
        r = eval_atom(r, env)
    if car(code) == "sumalista":
        pass
    if DEBUG: print("IEVAL: evaluate: ",stringify(code)," = ",stringify(r))
    return r

    
def ieval_args(args, env):
    if args == ():
        return ()
    # TODO comprimeir esto
    # TODO convertir en  no recursivo

    largs = pair_to_list(args)
    if args == []:
        return ()
    
    elargs = []
    for i in largs:
        ei = ieval(i,env)
        elargs += [ei]

    args = list_to_pair(elargs)

    return args

    # kar = ieval(car(args), env)


    # return (ieval(car(args), env), ieval_args(cdr(args), env))



def iapply_built_in(closure, args, env):
    e_args = ieval_args(args, env)
    if closure.name == "eval":
        return closure.body(e_args, env)
    if closure.name in ['car','cdr']:
        return closure.body(car(e_args))
    return closure.body(e_args)



def iapply(closure, args, env):

    if APPLY_DEBUG: print(f"IAPPLY: aplying: {closure.name} ",stringify(closure.body))

    new_env = closure.env + env

    if closure.built_in:
        return iapply_built_in(closure, args, env)
    
    # bind the parameters
    if APPLY_DEBUG: print("IAPPLY: unevaluated args: ", stringify(args))
    aux_args = ieval_args(args, env)
    aux_params = closure.params

    if APPLY_DEBUG: print("IAPPLY: aux_args = ",aux_args)
    if APPLY_DEBUG: print("IAPPLY: aux_params = ",aux_params)
    while aux_params != () and aux_args != ():
        # dot notation
        if car(aux_params) == ".":
            new_env[0][car(cdr(aux_params))] = aux_args
            aux_args = ()
            aux_params = ()
            break

        new_env[0][car(aux_params)] = car(aux_args)
        aux_params = cdr(aux_params)
        aux_args = cdr(aux_args)

        # if DEBUG: print("DEBUG: aux_args = ",aux_args)
        # if DEBUG: print("DEBUG: aux_params = ",aux_params)

    if aux_params != () or aux_args != ():
        merror("Mismatched number of arguments and parameters MAPPLY")
    
    r = ieval(closure.body, new_env)
    if APPLY_DEBUG: print("IAPPLY: closure = ",stringify(closure.body), " = ",r,"\n",
                    "\tparams = ",closure.params,"\n",
                    "\targs = ", args)
    return r

def eval_string(s,menv):
    # try:
    if True:
        t = tokenize(s)
        if DEBUG: print("DEBUG: EVAL_STRING tokens = ",t)
    # except Exception as e:
    #     print("Tokenization ERROR: ",e)
    #     return Error()
    if True:
        return block_evaluation(t)

def block_evaluation(tokens):
    aux = tokens.copy()
    try:
        p = parse(aux)
        p = expand_macros(p, menv)
        if DEBUG: print("EXPANDED: ",stringify(p))
        if DEBUG: print("DEBUG: block_evaluation: code = ",p)
        auxp = copy_pair(p)
        try:
            r = ieval(p,menv)
            print(stringify(r))
            return r
        except Exception as e:
            print("ERROR: ",e)
            print("ERROR: ",stringify(auxp))
            return Error()
    except Exception as e:
        print("Syntax Error: ",e)
        print("Syntax Error: "," ".join(aux))
        return Error()

# 
# ----    TOKENIZER AND PARSER    ----
#
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


     
def parse(tokens):
    if not tokens:
        return ()

    token = tokens.pop(0)
    if token not in [OPEN, CLOSE,"'","`",","]:
        return token
    
    # expansion de las quotes
    if token == "'":
        new_pair = ("quote",())
        new_pair = append(new_pair, parse(tokens))
        return new_pair
    
    if token == "`":
        new_pair = ("quasiquote",())
        new_pair = append(new_pair, parse(tokens))
        return new_pair
    
    if token == ",":
        new_pair = ("unquote",())
        new_pair = append(new_pair, parse(tokens))
        return new_pair
    
    if token == OPEN:
        if tokens[0] == CLOSE:
            tokens.pop(0)
            return ()
        kar = parse(tokens)
        new_pair = (kar, ())
        while tokens:
            if tokens and tokens[0] == CLOSE:
                tokens.pop(0)
                break
            new_pair = append(new_pair, parse(tokens))
        return new_pair
    return ()
        


# 
# ----    auxiliary    ----
#

def merror(message):
    raise Exception(message)

def stringify(code):
    c = ""
    if type(code) == tuple:
        if code == ():
            return "() "
        c += "( "
        c += stringify(car(code))
        # print all the args
        aux = cdr(code)
        while aux != None:
            if aux == ():
                break
            if type(aux) == tuple:
                c += " " + stringify(car(aux))
            else:
                c += " . " + stringify(aux)
                break
            aux = cdr(aux)
        c += ") "
        return c

    elif type(code) == Closure:
        c += "Procedure: <"
        c += stringify(code.name) + " "
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
    

def tuple_to_pair(t):
    if t == ():
        return ()
    return (t[0], tuple_to_pair(t[1:]))

def ttp(t):
    p = tuple_to_pair(t)
    if DEBUG: print("ttp = ",p)
    return p

def e(t, env):
    return ieval(ttp(t), env)

def printokens(tokens):
    return ' '.join(tokens)

def eval_file(file,menv):
    code = i_read_file(file)

    t = tokenize(code)
    if DEBUG: print("DEBUG: EVAL_FILE tokens = ",t)
    # print("EVAL_FILE: file = ",file)

    block = []
    parentesis_count = 0
    while t:
        token = t.pop(0)
        # print("TOKEN = ",token, " parentesis_count = ",parentesis_count)
        if token == "(":
            parentesis_count += 1
        if token == ")":
            parentesis_count -= 1
        block.append(token)
        if parentesis_count == 0:
            print("BLOCK = ",printokens(block))
            block_evaluation(block)
            block = []
            parentesis_count = 0
            continue



#
# ----    I/O    ----
#

#
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


#
# ----    TESTS    ----
#

def internal_test(env):

    print("TEST null")


    print("TEST, define lambda")
    eval_string("(define a (lambda (v) (+ v 1)))",menv)
    eval_string("(a 2)",menv)

    print("TEST cons")
    e = eval_string("(cons 1 2)",menv)
    e = eval_string("(cons 1 null)",menv)

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
    eval_string("(atom? 1)",menv)
    eval_string("(atom? (cons 1 2))",menv)
    eval_string("(atom? null)",menv)
    eval_string("(atom? (cons 1 null))",menv)
    eval_string("(atom? (lambda (x) (+ x x)))",menv)

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


    print("TEST, recursive lambda")
    eval_string("""
(define sumatorio (lambda (x)
                    (cond ((eq? x 0) 0)
                          (#t (+ x (sumatorio (- x 1)))))))
                """,menv)
    f = eval_string("(sumatorio 5)",menv)
    print("sumatorio 5 = ",stringify(f))


    print("TEST, other")




menv = [{}]
macros = {}


menv[0]["+"] = Closure(params = (), body = suma, built_in = True, name = "+")
menv[0]["-"] = Closure(params = (), body = resta, built_in = True, name = "-")
menv[0]["cons"] = Closure(params = (), body = cons, built_in = True, name = "cons")
menv[0]["car"] = Closure(params = (), body = car, built_in = True, name = "car")
menv[0]["cdr"] = Closure(params = (), body = cdr, built_in = True, name = "cdr")
menv[0]["atom?"] = Closure(params = (), body = atom_, built_in = True, name = "atom?")
menv[0]["eq?"] = Closure(params = (), body = eq_, built_in = True, name = "eq?")
menv[0]["eval"] = Closure(params = (), body = peval, built_in=True, name="eval")
menv[0]["display"] = Closure(params = (), body = display, built_in=True, name="display")
menv[0]["newline"] = Closure(params = (), body = newline, built_in=True, name="newline")
menv[0]["length"] = Closure(params = (), body = length, built_in=True, name="length")
menv[0]["pair?"] = Closure(params = (), body = pair_, built_in=True, name="pair?")
menv[0]["*"] = Closure(params = (), body = mult, built_in=True, name="*")
menv[0]["/"] = Closure(params = (), body = div, built_in=True, name="/")
menv[0][">"] = Closure(params = (), body = mayor_, built_in=True, name=">")
menv[0]["<"] = Closure(params = (), body = menor_, built_in=True, name="<")






# internal_test(menv)
# print(sys.argv)
print("type 'exit' for exit")

if len(sys.argv) == 2:
    eval_file(sys.argv[1],menv)

# a repl
while True:
    # use readline 
    code = input("one > ")
    if code == "exit":
        break
    if code != "":
        eval_string(code,menv)


