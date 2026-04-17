% ARES Octave state-space analysis helper.
% Computes controllability, observability, and closed-loop poles.

pkg load control;

A = [1.0, 0.1; 0.0, 0.95];
B = [0.0; 0.1];
C = [1.0, 0.0];
D = [0.0];
desired_poles = [0.55, 0.65];

K = place(A, B, desired_poles);
Acl = A - B * K;

ctrb_rank = rank(ctrb(A, B));
obsv_rank = rank(obsv(A, C));
poles = eig(Acl);

fid = fopen('results/control/octave_state_space_summary.txt', 'w');
fprintf(fid, 'Controllability rank: %d\n', ctrb_rank);
fprintf(fid, 'Observability rank: %d\n', obsv_rank);
fprintf(fid, 'Closed-loop poles: %.6f %.6f\n', poles(1), poles(2));
fclose(fid);

disp('Saved results/control/octave_state_space_summary.txt');
