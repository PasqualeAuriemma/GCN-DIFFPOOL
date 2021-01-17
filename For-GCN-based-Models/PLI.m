function matrix = PLI(Signal)
% Signal segnale su cui applicare il metodo

Signal = Signal';
% Numero canali
numChannels = size(Signal,2);

% inizializza la matrice finale con tutti zero
PLImatrix = zeros(numChannels, numChannels);

% Si ottengono gli angoli di fase applicando la trasformata di Hibelt.
% Attraverso la DFT si calcolano la rappresentazione analitica degli
% elementi del segnale passati come input. Dopo la trasformazione a numeri
% complessi di applica la funzione Angle per calcolarci l'angolo.
phaseSignal = angle(hilbert(Signal));

for i = 1:(numChannels-1)
    for m = (i+1):numChannels
        
        % differenza di fase tra il canle preso di riferimento e gli altri
        % canali
        diffPhase = phaseSignal(:,m)-phaseSignal(:,i);
        %figure(1);
        %subplot(221)
        %plot([Signal(:,i), Signal(:,m)])
        %subplot(223)
        %plot([phaseSignal(:,i), phaseSignal(:,m)])
        %subplot(224)
        %plot(diffPhase)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %t = diffPhase > 0;
        %count1 = sum(t);
        %p = abs(diffPhase) > 0;
        %count2 = sum(p);
        %plid = 2.0 * abs(0.5 - 1.0 * count1 / count2);
        %plot(t)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % calcola il segno degli elementi ottenuti dalla differenza di 
        % fase.
        signItem =  sign(diffPhase);
        %subplot(222)
        %plot(signItem)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %t = signItem > 0;
        %count1 = sum(t);
        %p = abs(signItem) > 0;
        %count2 = sum(p);
        %plid = 2.0 * abs(0.5 - 1.0 * count1 / count2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % calcola la media degli elementi ottenuti dal calcolo del segno.
        avarage = mean(signItem);
        % Calcolo valore assoluto delle medie
        PLImatrix(i,m) = abs(avarage);
        %PLImatrix(i,m) = plid;
    end
end

pli = PLImatrix;
pli = triu(pli);
pli = pli+pli';
pli(eye(size(pli))~=0)=1;
% Considera 1 - pli in modo tale da avere 0 quando due canali sono
% perfettamente simili 1 altrimenti.
matrix = pli;