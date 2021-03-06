t_0 = 0;
t_e = 8;
x_0 = 0;
x_e = 1;
x_steps = 20;
h = (x_e-x_0)/(x_steps-1);

x = h:h:x_e-h;
u_0 = exp(-x.^2);

u = ode23(@eq6, [t_0 t_e], u_0);
solution = u.y';

siz = size(solution);
t = linspace(t_0, t_e, siz(1))';
tau = t(2)-t(1);
solution_true = zeros(siz(1), x_steps-2);

for i = 1:siz(1)
    u_true = zeros(1, x_steps-2)';
    for j = 1:x_steps-2
        u_true(j) = exact(t(i), x(j));
    end
    solution_true(i, :) = u_true;
end

error =  sqrt(sum((solution-solution_true).^2, 2));

figure(2)
subplot(1, 2, 1)
[X,T]=meshgrid(x,t);
surf(T, X, solution, 'EdgeColor', 'interp');
xlabel('t')
ylabel('x')
zlabel('u')
title('Numerical solution')

subplot(1, 2, 2)
plot(t, error, '-');
xlabel('t')
ylabel('error')
title('Error(t)')
