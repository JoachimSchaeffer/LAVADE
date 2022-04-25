function obj = data_object_gui_interaction(app, dataset)
    switch dataset
        case 'Example'
            obj = Example(app);
        case 'ATR-FTIR Spectra'
            obj = Paracetamol();
        case 'LFP'
            obj = LFP();
        case 'Raman Spectra'
            obj = Raman(app.DropDownRun.Value, app.DropDownMeta.Value);
    end
end 


