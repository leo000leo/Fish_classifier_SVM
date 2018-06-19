function DisplayMatrix(confMat)
% Display the confusion matrix in a formatted table.

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

digits = 'a':'z';   %字符数组，输出种类名称
colHeadings = arrayfun(@(x)sprintf('%d',x),1:26,'UniformOutput',false);
format = repmat('%-6s',1,11);
header = sprintf(format,'Fish |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
% imagesc(format); 
% colormap(flipud(gray));
for idx = 1:numel(digits)
    fprintf('%-6s',   [digits(idx) '    |']);
    fprintf('%-6.3f', confMat(idx,:));
    fprintf('\n')
end