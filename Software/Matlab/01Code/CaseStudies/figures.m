%% File to reproduce/load the figures used in the paper

clc; close all; clear all; 
%% Reproduce the figures 
% The figures will not exactly look the same due to the random nature of
% the experiments

set(0,'defaultfigurecolor',[1 1 1])
%set('units','normalized','outerposition',[0 0 1 1])

%% Default Case

app = lavade_exported;

app.MethodDropDown.Value = "LASSO";
init_light(app);
plot_figures(app);
fig_coeff_lasso = figure;
fig_coeff_lasso.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_lasso);
hgsave(fig_coeff_lasso, 'Results/DC_LASSO_coeff.fig');
export_fig Results/DC_LASSO.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UIDC_LASSO.png');

app.MethodDropDown.Value = "PLS";
app.ComponentsEditField.Value = 2;
init_light(app);
plot_figures(app);
fig_coeff_pls = figure;
fig_coeff_pls.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_pls);
hgsave(fig_coeff_pls, 'Results/DC_PLS_coeff.fig');
export_fig Results/DC_PLS.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UIDC_PLS.png');

app.MethodDropDown.Value = "PCR";
app.ComponentsEditField.Value = 2;
init_light(app);
plot_figures(app);
fig_coeff_pcr = figure;
fig_coeff_pcr.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_pcr);
hgsave(fig_coeff_pcr, 'Results/DC_PCR_coeff.fig');
export_fig Results/DC_PCR.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UIDC_PCR.png');

app.MethodDropDown.Value = "RR";
app.RegularizationEditField.Value = 0.001;
init_light(app)
plot_figures(app);
fig_coeff_rr = figure;
fig_coeff_rr.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_rr);
hgsave(fig_coeff_rr, 'Results/DC_RR_coeff.fig');
export_fig Results/DC_RR.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UIDR_RR.png');

app.MethodDropDown.Value = "EN";
app.RegularizationEditField.Value = 0.001;
init_light(app)
plot_figures(app);
fig_coeff_en = figure;
fig_coeff_en.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_en);
hgsave(fig_coeff_en, 'Results/DC_EN_coeff.fig');
export_fig Results/DC_EN.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UIDC_EN.png');

%% DSC 
app.StandardizeInputsCheckBox.Value = true;

app.MethodDropDown.Value = "LASSO";
init_light(app);
plot_figures(app);
fig_coeff_lasso_dsc = figure;
fig_coeff_lasso_dsc.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_lasso_dsc);
hgsave(fig_coeff_lasso_dsc, 'Results/DSC_LASSO_coeff.fig');
export_fig Results/DSC_LASSO.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UIDSC_LASSO.png');

app.StandardizeInputsCheckBox.Value = false;

%% NC

app.SNRLeftSlider.Value = 20;
app.RightSlider.Value = 20;   
app.SignalSlider.Value = 20;
app.SigEEditField.Value = 2;
app.SigSEditField.Value = 2;
app.NoiseCheckBox.Value = true;


app.MethodDropDown.Value = "PLS";
app.ComponentsEditField.Value = 4;
app.HoldCheckBox.Value = false;
init(app)
fig_coeff_pls4_1_nc = figure;
fig_coeff_pls4_1_nc.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_pls4_1_nc);
hgsave(fig_coeff_pls4_1_nc, 'Results/NC_PLS4_1_coeff.fig');
export_fig Results/NC_PLS4_1.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UINC_PLS4_1.png');

app.HoldCheckBox.Value = true;
for i=1:5
    init(app)
end
fig_coeff_pls4_nc = figure;
fig_coeff_pls4_nc.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_pls4_nc);
hgsave(fig_coeff_pls4_nc, 'Results/NC_PLS4_coeff.fig');
export_fig Results/NC_PLS4_5.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UINC_PLS4_5.png');

app.ComponentsEditField.Value = 5;
app.HoldCheckBox.Value = false;
init(app)
fig_coeff_pls5_1_nc = figure;
fig_coeff_pls5_1_nc.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_pls5_1_nc);
hgsave(fig_coeff_pls5_1_nc, 'Results/NC_PLS5_1_coeff.fig');
export_fig Results/NC_PLS5_1.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UINC_PLS5_1.png');

app.HoldCheckBox.Value = true;
for i=1:5
    init(app)
end
fig_coeff_pls5_nc = figure;
fig_coeff_pls5_nc.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_pls5_nc);
hgsave(fig_coeff_pls5_nc, 'Results/NC_PLS5_coeff.fig');
export_fig Results/NC_PLS5_5.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UINC_PLS5_5.png');

app.MethodDropDown.Value = "RR";
app.RegularizationEditField.Value = 0.001;
app.HoldCheckBox.Value = false;
init_light(app)
plot_figures(app);
fig_coeff_rr0001_nc = figure;
fig_coeff_rr0001_nc.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_rr0001_nc);
hgsave(fig_coeff_rr0001_nc, 'Results/NC_RR0001_coeff.fig');
export_fig Results/NC_RR0001.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UINC_RR001.png');

app.HoldCheckBox.Value = true;
for i=1:10
    init(app)
end
fig_coeff_rr0001_10rep_nc = figure;
fig_coeff_rr0001_10rep_nc.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_rr0001_10rep_nc);
hgsave(fig_coeff_rr0001_10rep_nc, 'Results/NC_RR0001_10rep_coeff.fig');
export_fig Results/NC_RR0001_10.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UINC_RR0001_10.png');


app.RegularizationEditField.Value = 10;
app.HoldCheckBox.Value = false;
init(app)
app.HoldCheckBox.Value = true;
for i=1:10
    init(app)
end
fig_coeff_rr10_10rep_nc = figure;
fig_coeff_rr10_10rep_nc.WindowState = 'maximized';
copyobj(app.ax2, fig_coeff_rr10_10rep_nc);
hgsave(fig_coeff_rr10_10rep_nc, 'Results/NC_RR10_10rep_coeff.fig');
export_fig Results/NC_RR10_10.png -r500
%exportapp(app.LAVADEUIFigure, 'Results/UINC_RR10_10.png');


