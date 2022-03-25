function in =  localResetFcn(in)
% randomize initail theta1
theta1_0 = pi-0.1 + 0.2*rand;
in = in.setVariable('theta1_0',theta1_0);
end