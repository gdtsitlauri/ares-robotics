% ARES Octave PID step-response helper.
% Mirrors the lightweight first-order plant used in the Python baseline.

pkg load control;

s = tf('s');
plant = 1 / (s + 1.1);
controller = 2.4 + 0.8 / s + 0.12 * s;
closed_loop = feedback(controller * plant, 1);

t = 0:0.02:5;
[y, t] = step(closed_loop, t);

csvwrite('results/control/octave_pid_response.csv', [t, y]);
disp('Saved results/control/octave_pid_response.csv');
