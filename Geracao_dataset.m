clc;
clear;
rng('default')

j = 1;
k = 1;
for i=1:100
    while(1)
        A = rand(1,4);
        signal_a = randi([-1,1],1,4);
        signal_a(~signal_a) = 1;
        B = rand(1,3);
        signal_b = randi([-1,1],1,3);
        signal_b(~signal_b) = 1;
        num = [A(1,1)*signal_a(1,1) A(1,2)*signal_a(1,2) A(1,3)*signal_a(1,3) A(1,4)*signal_a(1,4)];
        den = [1 B(1,1)*signal_b(1,1) B(1,2)*signal_b(1,2) B(1,3)*signal_b(1,3)];
        funcoes_transferencia_1z_1p(1,i) = tf(num, den, 1);
        opt_2 = stepDataOptions('StepAmplitude',1);
        opt_1 = stepDataOptions('StepAmplitude',2);
        [Y] = [step(funcoes_transferencia_1z_1p(1,i))];
        [Y_imp] = [step(funcoes_transferencia_1z_1p(1,i))];
        X = ones(1, length(Y));
        X(1,1) = 0;
        t = 0:length(Y)-1;
        [Y] = lsim(funcoes_transferencia_1z_1p(1,i), X, t);
        if (abs(abs(Y(end)) - abs(Y(length(Y)-1))) < 10
            break
        end
    end
    respostas_ao_degrau(j:j+3,:) = [funcoes_transferencia_1z_1p.numerator(1,i);funcoes_transferencia_1z_1p.denominator(1,i);X;Y'];
    j = j+4;
    k = k+2;
end
writecell(respostas_ao_degrau,'Data_sets/Data_set4_3.csv', 'Delimiter', ';')
