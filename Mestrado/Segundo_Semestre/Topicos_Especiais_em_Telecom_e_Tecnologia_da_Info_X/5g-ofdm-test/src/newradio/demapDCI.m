function struct = demapDCI(dciBits,dciMask)
%DEMAPDCI Summary of this function goes here
%   Detailed explanation goes here
%   Detailed explanation goes here
% Bit size mask
struct.Identifier = 1;              % Always 1
struct.FrequencyResources = bi2de(dciBits(2:5)');      
struct.ResourceAssignment = bi2de(dciBits(6:9)');     
struct.VRBPRBMap = bi2de(dciBits(10)');               
struct.Modulation= bi2de(dciBits(11:15)');       
struct.NewData = bi2de(dciBits(16)');                
struct.RedundancyVersion = bi2de(dciBits(17:18)');      
struct.HARQ =  bi2de(dciBits(19:22)');                    
struct.DownlinkIDX = bi2de(dciBits(23:24)');               
struct.TPC = bi2de(dciBits(25:26)');         
struct.PUCCHResource = bi2de(dciBits(27:29)');
struct.PDSCHHARQ = bi2de(dciBits(30:32)');
end

