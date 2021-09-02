clc;
clear;
rng('default')

j = 1;
for i=1:100
    while(1)
        A = rand(1,4);
        signal_a = randi([-10,10],1,4);
        signal_a(~signal_a) = 1;
        B = rand(1,3);
        signal_b = randi([-10,10],1,3);
        signal_b(~signal_b) = 1;
        num = [A(1,1)*signal_a(1,1) A(1,2)*signal_a(1,2) A(1,3)*signal_a(1,3) A(1,4)*signal_a(1,4)];
        den = [1 B(1,1)*signal_b(1,1) B(1,2)*signal_b(1,2) B(1,3)*signal_b(1,3)];
        funcoes_transferencia_1z_1p(1,i) = tf(num, den, 1);
        opt_1 = stepDataOptions('StepAmplitude',1);
        opt_2 = stepDataOptions('StepAmplitude',2);
        opt_3 = stepDataOptions('StepAmplitude',-3);
        [Y] = [step(funcoes_transferencia_1z_1p(1,i), opt_1)];
        [Y_2A] = [step(funcoes_transferencia_1z_1p(1,i), opt_2)];
        [Y_3A] = [step(funcoes_transferencia_1z_1p(1,i), opt_3)];
        if (length(Y) > 120 || length(Y_2A) > 120 || length(Y_3A) > 120)
            continue
        end
        X = ones(1, length(Y));
        X_2A = 2*ones(1, length(Y_2A));
        X_3A = -3*ones(1, length(Y_2A));
        X_tri = zeros(1, 2*length(Y));
        X(1,1) = 0;
        X_2A(1,1) = 0;
        X_3A(1,1) = 0;
        X_tri(1,1) = 0;X_tri(1,2:6) = 1;
        t = 0:length(Y)-1;
        t_2A = 0:length(Y_2A)-1;
        t_3A = 0:length(Y_3A)-1;
        t_tri = 0:2*length(Y)-1;
        [Y] = lsim(funcoes_transferencia_1z_1p(1,i), X, t);
        [Y_2A] = lsim(funcoes_transferencia_1z_1p(1,i), X_2A, t_2A);
        [Y_3A] = lsim(funcoes_transferencia_1z_1p(1,i), X_3A, t_3A);
        [Y_tri] = lsim(funcoes_transferencia_1z_1p(1,i), X_tri, t_tri);
        if (abs(abs(Y(end)) - abs(Y(length(Y)-1))) < 10 && abs(abs(Y_2A(end)) - abs(Y_2A(length(Y_2A)-1))) < 10 ...
            && abs(abs(Y_3A(end)) - abs(Y_3A(length(Y_3A)-1))) < 10 && abs(abs(Y_tri(end)) - abs(Y_tri(length(Y_tri)-1))) < 10)
            break
        end
    end
    respostas_ao_degrau(j:j+9,:) = [funcoes_transferencia_1z_1p.numerator(1,i);funcoes_transferencia_1z_1p.denominator(1,i);X;Y';
                                    X_2A;Y_2A';X_3A;Y_3A';X_tri;Y_tri'];
    j = j+10;
end
writecell(respostas_ao_degrau,'Data_sets/Data_set_10_4_3_tst.csv', 'Delimiter', ';')