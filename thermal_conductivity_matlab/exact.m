function u = exact(t,x)
    u = 1-cos(t)+exp((-x.^2)/(1+4*t))/sqrt(1+4*t);
    