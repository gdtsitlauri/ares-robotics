default rel

section .text
global simulate_interrupt_ticks

; rdi = tick_count
; returns rax = accumulated ticks
simulate_interrupt_ticks:
    xor rax, rax
    xor rcx, rcx
.loop:
    cmp rcx, rdi
    jge .done
    inc rax
    inc rcx
    jmp .loop
.done:
    ret
