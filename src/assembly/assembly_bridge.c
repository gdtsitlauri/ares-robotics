#include <math.h>

extern double pid_update(double error, double integral, double prev_error, double kp, double ki, double kd);

double assembly_pid(double error, double integral, double prev_error, double kp, double ki, double kd) {
    return pid_update(error, integral, prev_error, kp, ki, kd);
}

double c_pid(double error, double integral, double prev_error, double kp, double ki, double kd) {
    return kp * error + ki * integral + kd * (error - prev_error);
}
