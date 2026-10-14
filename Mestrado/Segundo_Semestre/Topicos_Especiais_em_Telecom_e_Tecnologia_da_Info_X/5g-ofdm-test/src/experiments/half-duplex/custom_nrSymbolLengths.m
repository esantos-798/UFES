function cp = custom_nrSymbolLengths(carrier, info, L)
    % Comprimento do CP (amostras) dos L primeiros simbolos, a partir de carrier.NSlot
    nPerSub = numel(info.CyclicPrefixLengths);
    slotInSub = mod(carrier.NSlot, info.SlotsPerSubframe);
    idx = mod(slotInSub * 14 + (0:L-1), nPerSub) + 1;
    cp = info.CyclicPrefixLengths(idx);
end
