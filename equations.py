def exponent (x:float) -> float:
    mult = 1
    m = 0
    n = 1
    w = 1
    z = 0
    y = 1
    c =1
    mult2 =1
    while n < 30:
        while m < y:
            mult *= x
            m+=1
        while z < y:
            mult2 *= c
            z +=1
            c += 1
        w += mult/mult2
        n+=1
        y+=1
        m = 0
        mult = 1
        c = 1
        mult2 = 1
        z = 0
    return w
def Ln(x:float) -> float:
    yn = x - 1.0
    yn1 = 0.0
    epsilon = 0.001
    s = yn-yn1
    while s > epsilon:
        yn1 = yn + (2*((x-exponent(yn))/(x+exponent(yn))))
        s = yn1 - yn
        yn = yn1
        if s < 0:
            s = -s
    return yn1
def XtimesY (x:float, y:float) -> float:
    if x <= 0:
        return 0
    z =  float('%0.6f' % exponent(y*Ln(x)))
    return z
def sqrt(x:float,y:float):
    z = XtimesY (x,1/y)
    return z
def calculate(x:float) -> float:
    z = exponent(x)*XtimesY(7,x)*XtimesY(x,-1)*sqrt(x,x)
    return z
    
    
    



    
    
    
    
   
