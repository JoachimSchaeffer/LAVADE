function obj = data_object_gui_interaction(app, dataset)
    switch dataset
        case 'Example'
            obj = Example(app);
        case 'Paracetamol'
            obj = Paracetamol();
        case 'Paracetamol_original'
            obj = Paracetamol_original();
        case 'LFP'
            obj = LFP();
        case 'Raman'
            obj = Raman(app.DropDownRun.Value, app.DropDownMeta.Value);
    end
end 


