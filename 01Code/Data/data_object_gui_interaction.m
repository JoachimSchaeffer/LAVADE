function obj = data_object_gui_interaction(app, dataset)
    switch dataset
        case 'Example'
            obj = Example(app);
        case 'Paracetamol'
            obj = Paracetamol();
        case 'LFP'
            obj = LFP();
    end
end 


