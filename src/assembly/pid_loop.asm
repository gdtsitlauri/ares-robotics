default rel

section .text
global pid_update

; System V AMD64 ABI
; xmm0 = error
; xmm1 = integral
; xmm2 = prev_error
; xmm3 = kp
; xmm4 = ki
; xmm5 = kd
; return xmm0 = kp*error + ki*integral + kd*(error-prev_error)
pid_update:
    movapd xmm6, xmm0
    subsd xmm6, xmm2
    mulsd xmm6, xmm5

    mulsd xmm0, xmm3
    mulsd xmm1, xmm4
    addsd xmm0, xmm1
    addsd xmm0, xmm6
    ret

section .note.GNU-stack noalloc noexec nowrite progbits
