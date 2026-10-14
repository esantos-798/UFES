function [DLgrid, waveform, cInfo, pdschSymbols, trBlk, txInfo] = ...
    custom_downlink_tx(cCarrier, pSizeGrid, pDigitalModulation, pNCellID)
    % custom_downlink_tx Emula o TX do main_downlink.m SEM 5G Toolbox.
    %
    % Grade de 2 slots (28 simbolos), conforme initSSB.m / mapPDCCH.m / mapPDSCH.m / mapDLGrid.m do laboratorio:
    %   - simbolos 2..5 : SSB (240 subportadoras a partir de round((K-240)/2)+1):
    %                     PSS e SSS reais; PBCH com DMRS em k = mod(NCellID,4) (mod 4) e dados QPSK aleatorios
    %   - simbolo 7     : PDCCH (24 RB; DMRS em k = 1,5,9 de cada RB; 216 RE de dados QPSK aleatorios)
    %   - simbolos 15..28 (slot 2): PDSCH, DMRS tipo 1 no simbolo 3 do slot (subportadoras pares,
    %                     sequencia Gold com NID = cCarrier.NCellID), simbolo de DMRS sem dados
    %                     (NumCDMGroupsWithoutData = 2)
    % Diferencas vs. toolbox: sem codificacao LDPC/polar, sem scrambling dos dados.
    % trBlk = bits brutos mapeados no PDSCH (BER sera "pre-FEC"). txInfo.trBlkSize informa o
    % tamanho do bloco de transporte que o DL-SCH real usaria (custom_nrTBS, conforme initDLSCH.m).

    switch upper(pDigitalModulation)
        case 'QPSK',  M = 4;
        case '16QAM', M = 16;
        case '64QAM', M = 64;
        otherwise
            error('custom_downlink_tx:Mod', 'Modulacao nao suportada: %s', pDigitalModulation);
    end
    k = log2(M);
    K = 12 * pSizeGrid;
    if K < 240
        error('custom_downlink_tx:Grid', 'NSizeGrid >= 20 e necessario para o SSB.');
    end
    L = 28;
    DLgrid = zeros(K, L);
    qpsk = @(n) ((1 - 2*randi([0 1], n, 1)) + 1i*(1 - 2*randi([0 1], n, 1))) / sqrt(2);

    % PDCCH (AggregationLevel 4, CORESET 1 simbolo, 24 RB a partir do RB 0; mapPDCCH: nID = 23,
    % E = 432 bits = 216 RE de dados + 72 RE de DMRS nas subportadoras 1, 5, 9 de cada RB).
    % mapDLGrid.m coloca o simbolo do PDCCH na coluna 7 da grade (grid(:,7) = pdcchGrid(:,1)).
    nPdcch = min(K, 24*12);
    kk = (0:nPdcch-1).';
    isDmrs = (mod(kk, 4) == 1);
    cinitP = mod(2^17 * (14*0 + 0 + 1) * (2*23 + 1) + 2*23, 2^31);
    cP = goldSeq(cinitP, 2*sum(isDmrs));
    pdcchDmrs = (((1 - 2*cP(1:2:end)) + 1i*(1 - 2*cP(2:2:end))) / sqrt(2));
    col7 = zeros(nPdcch, 1);
    col7(isDmrs) = pdcchDmrs(:);
    col7(~isDmrs) = qpsk(sum(~isDmrs));
    DLgrid(1:nPdcch, 7) = col7;

    % SSB (initSSB.m): bloco 240 x 4 posto em DLgrid(offset:offset+239, 2:5), offset = round((K-240)/2)+1
    NID2 = mod(pNCellID, 3);
    NID1 = floor(pNCellID / 3);
    ssb = zeros(240, 4);
    ssb(57:183, 1) = custom_nrPSS(NID2);                 % PSS: simbolo 0 do SSB, k = 56..182
    ssb(57:183, 3) = custom_nrSSS(NID1, NID2);           % SSS: simbolo 2 do SSB
    pbch = false(240, 4);
    pbch(:, 2) = true;                                   % PBCH: simbolo 1 inteiro
    pbch([1:48, 193:240], 3) = true;                     % PBCH: simbolo 2, k = 0..47 e 192..239
    pbch(:, 4) = true;                                   % PBCH: simbolo 3 inteiro
    v = mod(pNCellID, 4);
    kSSB = repmat((0:239).', 1, 4);
    pbchDmrsMask = pbch & (mod(kSSB, 4) == v);           % DMRS do PBCH em k mod 4 = v (144 RE)
    pbchDataMask = pbch & ~pbchDmrsMask;                 % 432 RE de dados
    cinitB = 2^11 * (0+1) * (floor(pNCellID/4)+1) + 2^6 * (0+1) + mod(pNCellID, 4);   % ibar_SSB = 0
    cB = goldSeq(cinitB, 2*sum(pbchDmrsMask(:)));
    ssb(pbchDmrsMask) = (((1 - 2*cB(1:2:end)) + 1i*(1 - 2*cB(2:2:end))) / sqrt(2));
    ssb(pbchDataMask) = qpsk(sum(pbchDataMask(:)));
    offset = round((K - 240) / 2) + 1;
    DLgrid(offset:offset+239, 2:5) = ssb;

    % PDSCH + DMRS (grade do slot: K x 14)
    dmrsRows = (1:2:K).';
    dmrsIndices = dmrsRows + (3-1)*K;               % simbolo 3 do slot, indices do slot
    NIDc = 1;                                       % default do nrCarrierConfig (o main nao ajusta NCellID)
    if isstruct(cCarrier) && isfield(cCarrier, 'NCellID'), NIDc = cCarrier.NCellID; end
    cinit = mod(2^17 * (14*0 + 2 + 1) * (2*NIDc + 1) + 2*NIDc, 2^31);
    c = goldSeq(cinit, 2*numel(dmrsRows));
    dmrsSymbols = ((1 - 2*c(1:2:end)) + 1i*(1 - 2*c(2:2:end))).' / sqrt(2);
    dmrsSymbols = dmrsSymbols(:);

    mask = true(K, 14);
    mask(:, 3) = false;
    pdschIndices = find(mask(:));

    trBlk = randi([0 1], numel(pdschIndices) * k, 1);
    pdschSymbols = custom_nrSymbolModulate(trBlk, M);

    slotGrid = zeros(K, 14);
    slotGrid(dmrsIndices) = dmrsSymbols;
    slotGrid(pdschIndices) = pdschSymbols;
    DLgrid(:, 15:28) = slotGrid;

    txInfo.NCellID = pNCellID;
    txInfo.M = M;
    txInfo.pdschIndices = pdschIndices;             % relativos ao slot (K x 14)
    txInfo.pdschSymbols = pdschSymbols;
    txInfo.dmrsIndices = dmrsIndices;
    txInfo.dmrsSymbols = dmrsSymbols;
    % Informativo (nao usado na cadeia): bloco de transporte do DL-SCH real (initDLSCH.m)
    txInfo.pdschG = numel(pdschIndices) * k;
    txInfo.trBlkSize = custom_nrTBS(pDigitalModulation, 1, pSizeGrid, numel(pdschIndices)/pSizeGrid, 490/1024, 0);

    [waveform, cInfo] = custom_nrOFDMModulate(cCarrier, DLgrid);
end

function c = goldSeq(cinit, len)
    % Sequencia Gold de comprimento 31 (TS 38.211 secao 5.2.1)
    Nc = 1600;
    N = Nc + len + 31;
    x1 = zeros(1, N); x2 = zeros(1, N);
    x1(1) = 1;
    x2(1:31) = double(bitget(uint32(cinit), 1:31));
    for n = 1:N-31
        x1(n+31) = mod(x1(n+3) + x1(n), 2);
        x2(n+31) = mod(x2(n+3) + x2(n+2) + x2(n+1) + x2(n), 2);
    end
    c = mod(x1(Nc+1:Nc+len) + x2(Nc+1:Nc+len), 2);
end
