function [data, header] = mireadsingle(file)

f = fopen (file,'r');

if f == -1
    data = [];
    header = [];
    return
end

buf=fread(f, 'uchar=>uint8')';

ind = findstr(buf, 'data');
cut = find(buf(ind:end)==10,1,'first');

c = textscan(char(buf(1:ind+cut-1))', '%14s%s','Delimiter','\n','Whitespace','');

header = makestruct(c);

nbufs = size(header.bufferLabel,1)-1;

if strcmpi(header.fileType,'Spectroscopy')
    if header.DataPoints == 0
        data = zeros(0,3,nbufs);
        return
    end
    
    if strcmpi(header.data, 'BINARY')
        %data = convchars2float(buf(ind+cut:end));               
        data = convchar(buf(ind+cut:end),'single');
        nd = numel(data);
        
        pts=[0;header.chunk(:,2)];
        cpts=cumsum(pts);
        
        if nd ~= cpts(end) || nd~= header.DataPoints
            data = nan;
            header = nan;
        end

        t=zeros(cpts(end),1);
        x=t;
        
        for k=1:length(pts) - 1 
            ind = cpts(k) + (1:pts(k+1));
            t(ind) = header.chunk(k,3) + (0:header.chunk(k,2)-1)*header.chunk(k,4);
            x(ind) = header.chunk(k,5) + (0:header.chunk(k,2)-1)*header.chunk(k,6);
        end
        
        data = reshape (data, [nd/nbufs, 1, nbufs]);
        data = cat(2, repmat(x, [1, 1, nbufs]), data, repmat(t, [1,1,nbufs]));        
    elseif strcmpi(header.data, 'ASCII')
        c = textscan(char(buf(ind+cut:end))','%n%n%n');
        t = reshape(c{1}, [numel(c{1})/nbufs, 1, nbufs]);
        x = reshape(c{2}, [numel(c{2})/nbufs, 1, nbufs]);
        data = reshape(c{3}, [numel(c{3})/nbufs, 1, nbufs]);
        data = cat(2, x, data, t);
    else
        data = nan;
        header = nan;
        return
    end    
elseif strcmpi(header.fileType,'Image')
    if strcmpi(header.data,'BINARY')
        % data = convchars2int32(buf(ind+cut:end));
        data = convchar(buf(ind+cut:end), 'int16');
    elseif strcmpi(header.data,'ASCII')
        data = textscan (char(buf(ind+cut:end))','%n');
        data = data{1};
    else
        data = nan;
        header = nan;
        return;
    end
    
    data = reshape(data, header.xPixels, header.yPixels, []);
    data = permute (data, [2 1 3]);
    data = data(end:-1:1,:,:);
end

fclose(f);
    
function header=makestruct(c)

header=struct;
for k=1:length(c{1})
    if strncmpi(c{1}{k},'color',5)
        continue
    end
    tmp=str2num(c{2}{k}); %#ok<ST2NM>
    if isempty(tmp)
        tmp=deblank(c{2}{k});
    end
    field = deblank(c{1}{k});
    if ~isfield(header, field)
        header.(field)=tmp;
    else
        if ischar(tmp)
            header.(field)=strvcat(header.(field), tmp); %#ok<VCAT>
        else
            header.(field)=cat(1, header.(field), tmp);
        end
    end
end
