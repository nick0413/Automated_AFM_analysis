function [data, header, files] = miread (file)
% This functions loads data saved with PicoView. It can read both,
% Spectroscop and Image files.
% [data, header, files] = miread (file)
% INPUT: file: filename of the file to load. Multiple file load can be done 
%              using wildcards and/or using a cell array of strings with
%              the files to load.
% OUTPUT: data : Cell array of the raw data of the according files. 
%                The Image of the lth file of the kth buffer can be converted
%                to its according unit by:
%                image = data{l}(:,:,k)/32768 * header{l}.bufferRange(k);
%         header : Cell array of structures containing the header of the
%                  file
%         files  : Cell array containing the names of the loaded files.


narg = nargin;

if narg < 1
    [file,pfad]=uigetfile('*.*', 'Select file(s) to load',...
        'MultiSelect','on');
    if iscell(file)
        file = cellfun(@(x)fullfile(pfad, x), file, 'UniformOutput', false);
    else
        file={fullfile(pfad, file)};
    end
end

if ~iscell(file)
    file = {file};
end

files=cell(0,1);
h=waitbar(0,'Initializing...');
set(get(get(h,'Children'),'Title'),'Interpreter','none');

for k=1:length(file)
    ind = find(file{k}==filesep,1,'last');
    s=dir(file{k});
    if isempty(s)
        continue;
    end
    tmp={s.name};
    tmp=cellfun(@(x)fullfile(file{k}(1:ind), x), tmp, 'UniformOutput', false);
    files=cat(1,files, tmp);
end

lf = length(files);

data = cell(lf,1);
header = cell(lf,1);

for k=1:lf    
    ind = find(files{k}==filesep,1,'last');
    waitbar(k/lf, h, ['Lade ',files{k}(ind+1:end),'...']);
    [data{k}, header{k}] = mireadsingle(files{k});    
end

close(h);

    
