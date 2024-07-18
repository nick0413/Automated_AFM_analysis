import originpro as op

# Open a new Origin instance   
app = op.Application()

# Open a new workbook
wb = app.new_book()

# Import data from a file
wb.from_file('data.txt')

# Perform some data processing
wb.active_layer.do_something()

# Save the workbook
wb.save('processed_data.opj')
