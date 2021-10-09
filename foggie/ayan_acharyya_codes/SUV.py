##!/usr/bin/env python3

"""

    Title :      SUV (Stupid Useless Visualisation)
    Notes :      Very basic attempts to play around with a GUI that'd display projection, 3D renderings and 2D plots from FOGGIE galaxies
                 This code is built upon manukalia's code on github https://github.com/manukalia/caiso_day-ahead_price_fetch/blob/master/utility_caiso_da_price_fetch.py
    Output :     should be able to save plots as png files or movies as mp4, once development is complete
    Author :     Ayan Acharyya
    Started :    October 2021
    Example :    python SUV.py

"""
from header import *
from util import *
from datashader_movie import *
import PySimpleGUI as sg
from types import SimpleNamespace
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")

start_time = time.time()
yt_ver = yt.__version__

# ---------------------------------------------
def draw_figure(canvas, figure):
    '''
    Function to draw a figure on PySimpleGUI's Canvas
    '''
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return figure_canvas_agg

# -----------------------------------------------------------------
def refresh_plot(box, window, args):
    '''
    Function to re-make datashader plot base don the new args
    '''
    myprint('Refreshing plot with galrad=%2F, xcol = %s, ycol = %s, colorcol = %s' % (args.galrad, args.ycolname, args.xcolname, args.colorcolname), args)
    inflow_outflow_text = ''  ## temporary
    args.current_redshift = ds.current_redshift
    args.current_time = ds.current_time.in_units('Gyr')

    outfile_rootname = 'datashader_boxrad_%.2Fkpc_%s_vs_%s_colby_%s%s.png' % (args.galrad, args.ycolname, args.xcolname, args.colorcolname, inflow_outflow_text)
    thisfilename = figs_pathname + outfile_rootname

    df = get_df_from_ds(box, args)
    df, fig = make_datashader_plot_mpl(df, thisfilename, args)
    draw_figure(window['canvas'].TKCanvas, fig)

    return fig

# ---------------------------------------------------------
def update_galrad(ds, vis_window, args):
    '''
    Function to update the radius up to which computation will be performed,
    and cuts out smaller box accordingly,
    and updates the interactive window
    '''
    print('Updating galrad parameter..')
    args.galrad = float(args.galrad)
    # ----------further truncating dataset if necessary-----------------
    if args.fullbox:
        box_width = ds.refine_width  # kpc
        args.galrad = box_width / 2
        box = refine_box
        vis_window['galrad'].update('%.2F' % args.galrad)
    else:
        box_center = ds.arr(args.halo_center, kpc)
        box_width = args.galrad * 2  # in kpc
        box_width_kpc = ds.arr(box_width, 'kpc')
        box = ds.r[box_center[0] - box_width_kpc / 2.: box_center[0] + box_width_kpc / 2.,
              box_center[1] - box_width_kpc / 2.: box_center[1] + box_width_kpc / 2.,
              box_center[2] - box_width_kpc / 2.: box_center[2] + box_width_kpc / 2., ]

    return box, box_width, args

# ---------------------------------------------------------
def update_xparams(values_input_dict, vis_window, args):
    '''
    Function to update column name, min, max for x axis quantities, and updates the interactive window
    '''
    print('Updating x-axis parameters..')
    args.xcol = values_input_dict['xcol']
    args.xcolname = 'log_' + args.xcol if islog_dict[args.xcol] else args.xcol
    if isfield_weighted_dict[args.xcol] and args.weight: args.xcolname += '_wtby_' + args.weight
    args.xmin = np.log10(bounds_dict[args.xcol][0]) if islog_dict[args.xcol] else bounds_dict[args.xcol][0] if \
    values_input_dict['xmin'] == '' else float(values_input_dict['xmin'])
    args.xmax = np.log10(bounds_dict[args.xcol][1]) if islog_dict[args.xcol] else bounds_dict[args.xcol][1] if \
    values_input_dict['xmax'] == '' else float(values_input_dict['xmax'])

    vis_window['xmin'].update(args.xmin)
    vis_window['xmax'].update(args.xmax)

    return args

# ---------------------------------------------------------
def update_yparams(values_input_dict, vis_window, args):
    '''
    Function to update column name, min, max for y axis quantities, and updates the interactive window
    '''
    print('Updating y-axis parameters..')
    args.ycol = values_input_dict['ycol']
    args.ycolname = 'log_' + args.ycol if islog_dict[args.ycol] else args.ycol
    if isfield_weighted_dict[args.ycol] and args.weight: args.ycolname += '_wtby_' + args.weight
    args.ymin = np.log10(bounds_dict[args.ycol][0]) if islog_dict[args.ycol] else bounds_dict[args.ycol][0] if \
    values_input_dict['ymin'] == '' else float(values_input_dict['ymin'])
    args.ymax = np.log10(bounds_dict[args.ycol][1]) if islog_dict[args.ycol] else bounds_dict[args.ycol][1] if \
    values_input_dict['ymax'] == '' else float(values_input_dict['ymax'])

    vis_window['ymin'].update(args.ymin)
    vis_window['ymax'].update(args.ymax)

    return args

# ---------------------------------------------------------
def update_cparams(values_input_dict, vis_window, args):
    '''
    Function to update column name, min, max for color axis quantities, and updates the interactive window
    '''
    print('Updating color axis parameters..')
    args.colorcol = values_input_dict['colorcol']
    args.colorcolname = 'log_' + args.colorcol if islog_dict[args.colorcol] else args.colorcol
    args.colorcol_cat = 'cat_' + args.colorcolname
    if isfield_weighted_dict[args.colorcol] and args.weight: args.colorcolname += '_wtby_' + args.weight
    args.cmin = np.log10(bounds_dict[args.colorcol][0]) if islog_dict[args.colorcol] else bounds_dict[args.colorcol][0] if \
    values_input_dict['cmin'] == '' else float(values_input_dict['cmin'])
    args.cmax = np.log10(bounds_dict[args.colorcol][1]) if islog_dict[args.colorcol] else bounds_dict[args.colorcol][1] if \
    values_input_dict['cmax'] == '' else float(values_input_dict['cmax'])
    if args.cmap == '': args.cmap = None

    vis_window['cmin'].update(args.cmin)

    return args

# -----main code-----------------
if __name__ == '__main__':
    # -------USER INPUTS WINDOW LOOP-----------
    fontname, fontsize = 'Raleway', 14
    sg.ChangeLookAndFeel('GreenMono')

    input_window_layout = [[sg.Text('\nChoose the appropriate paths and output(s) to load, and then hit SUBMIT to load the data..', font=(fontname, 18))],
                            [sg.Text('Choose source code location:', font=(fontname, fontsize))],
                            [sg.InputText('/Users/acharyya/Work/astro/ayan_codes/foggie/foggie', key='code_path', size=(55, 1), font=(fontname, fontsize)), sg.FolderBrowse(target='code_path')],
                            [sg.Text('Choose simulation data location:', font=(fontname, fontsize))],
                            [sg.InputText('/Users/acharyya/models/simulation_output/foggie', key='foggie_dir', enable_events=True, size=(55, 1), font=(fontname, fontsize)), sg.FolderBrowse(target='foggie_dir')],
                            [sg.Text('Choose filesave location:', font=(fontname, fontsize))],
                            [sg.InputText('/Users/acharyya/Work/astro/foggie_outputs', key='output_dir', size=(55, 1), font=(fontname, fontsize)), sg.FolderBrowse(target='output_dir')],
                            [sg.Text('Which machine are you on?', font=(fontname, fontsize))],
                            [sg.Combo(['ayan_local', 'ayan_hd', 'ayan_pleiades'], default_value='ayan_local', key='system', font=(fontname, fontsize))],
                            [sg.Text('\n', font=(fontname, 6))],
                            [sg.Frame(layout=[[sg.Checkbox('Over-write files if existing', size=(25,1), font=(fontname, fontsize), default=False, key='clobber'),
                                            sg.Checkbox('Save plots', size=(25,1), font=(fontname, fontsize), key='saveplot', default=False)], \
                                              [sg.Checkbox('Suppress all print messages', size=(25,1), font=(fontname, fontsize), key='silent', default=False)]],
                                            title='CHOOSE OPTIONS', font=(fontname, 18), title_color='darkblue', relief=sg.RELIEF_SUNKEN)],
                            [sg.Text('\n', font=(fontname, 6))],
                            [sg.Text('_' * 80)],
                            [sg.Text('\nChoose a halo name', font=(fontname, 18))],
                            [sg.Combo(['Tempest (8508)', 'Blizzard (4123)'], default_value='Tempest (8508)', key='halo', enable_events=True, font=(fontname, fontsize))],
                            [sg.Text('Choose a run name', font=(fontname, 18))],
                            [sg.Combo(['nref11c_nref9f'], default_value='nref11c_nref9f', key='run', enable_events=True, font=(fontname, fontsize))],
                            [sg.Text('Choose a snapshot', font=(fontname, 18))],
                            [sg.Combo(['RD0042', 'RD0038'], default_value='RD0042', key='output', font=(fontname, fontsize))],
                            [sg.Submit(), sg.Cancel()]]

    input_window = sg.Window('Preamble to FOGGIE Visualisation Utility', input_window_layout, resizable=True, element_justification='center', location=(180, 120), default_element_size=(20, 1), grab_anywhere=True)

    # -------updating user input window dynamically--------------
    print('Deb168: input window ready to accept changes') #
    while True:
        event, user_input_dict = input_window.read()
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            input_window.close()
            sg.popup_error('Operation cancelled')
            sys.exit(69)

        if event == 'Submit':
            input_window.close()
            break

        if event == 'foggie_dir': # if a foggie directory name was filled in, make a list of available halo names in that directory
            foggie_dir = user_input_dict['foggie_dir']
            halo_dirs = glob.glob(foggie_dir +'/halo_00*/')
            halo_names = [os.path.normpath(item).split(os.path.sep)[-1][-4:] for item in halo_dirs]
            halo_names_to_display = [halo_dict[item] + '(' + item + ')' for item in halo_names]
            input_window['halo'].update(value='', values=halo_names_to_display)

        elif event == 'halo': # if a halo name was filled in, make a list of available run names for that halo
            haloname = user_input_dict['halo'].split('(')[1][:-1]
            foggie_dir = user_input_dict['foggie_dir'] + '/halo_00' + haloname
            run_dirs = glob.glob(foggie_dir +'/nref*/')
            run_names = [os.path.normpath(item).split(os.path.sep)[-1] for item in run_dirs]
            input_window['run'].update(value='', values=run_names)

        elif event == 'run': # if a run name was filled in, make a list of available snapshots for that run
            haloname = user_input_dict['halo'].split('(')[1][:-1]
            run = user_input_dict['run']
            foggie_dir = user_input_dict['foggie_dir'] + '/halo_00' + haloname + '/' + run
            output_dirs = glob.glob(foggie_dir +'/RD*/') + glob.glob(foggie_dir +'/DD*/')
            output_names = [os.path.normpath(item).split(os.path.sep)[-1] for item in output_dirs]
            input_window['output'].update(value='', values=output_names)


    # -------new window to confirm user input-------------------
    sg.PopupScrolled('\n\nWindow auto closes in 10 sec (or hit OK to close)\n\n',
                     'You entered the following parameters:\n',
                     f'Halo name:  {user_input_dict["halo"]}',
                     f'Snapshot:     {user_input_dict["output"]}',
                     f'Run name:      {user_input_dict["run"]}',
                     f'System:      {user_input_dict["system"]}\n',
                     f'Over-write files if existing:     {user_input_dict["clobber"]}',
                     f'Save plots:    {user_input_dict["saveplot"]}',
                     f'Silent all print messages:    {user_input_dict["silent"]}\n',
                     f'Source code location:  {user_input_dict["code_path"]}',
                     f'Simulation data location:  {user_input_dict["foggie_dir"]}',
                     f'Output folder:  {user_input_dict["output_dir"]}',
                     font=(fontname, 18), title='', size=(80, 20), location=(180, 120), auto_close=True, auto_close_duration=10)

    # -------read in user inputs to args-------------------
    args = SimpleNamespace()
    for key in user_input_dict.keys():
        setattr(args, key, user_input_dict[key])

    args.print_to_file = False
    args.pwd = False
    args.halo = args.halo.split('(')[1][:-1]
    args.output_dir = user_input_dict['output_dir'] + '/plots_halo_00' + args.halo + '/' + args.run
    args.foggie_dir += '/'
    args.code_path += '/'
    args.output_dir += '/'
    print('\nDeb128:', args) #

    args = pull_halo_center(args)  # pull details about center of the snapshot
    if type(args) is tuple: args, ds, refine_box = args

    # ------------create directories if necessary---------------------------
    tables_pathname = args.output_dir + 'txtfiles/' + args.output + '/'
    figs_pathname = args.output_dir + 'figs/' + args.output + '/'

    Path(tables_pathname).mkdir(parents=True, exist_ok=True)
    Path(figs_pathname).mkdir(parents=True, exist_ok=True)

    # -------------load dataset based on args, create new fields------------------------
    ds, refine_box = load_sim(args, region='refine_box', do_filter_particles=False, disk_relative=False)
    ds.add_field(('gas', 'angular_momentum_phi'), function=phi_angular_momentum, sampling_type='cell', units='degree')
    ds.add_field(('gas', 'angular_momentum_theta'), function=theta_angular_momentum, sampling_type='cell', units='degree')

    particle_type, quantity = np.transpose(ds.derived_field_list)
    new_quantities = [quantity[index] for index in range(len(quantity)) if particle_type[index] == 'gas']
    existing_quantities = ['rad', 'vrad', 'temp', 'density', 'metal', 'phi_L', 'theta_L', 'phi_disk', 'theta_disk'] # parameters (like min, max, cmap) for these quantities already exist in the code
    all_quantities = existing_quantities + new_quantities

    # -------------new window for visualisation---------------------------
    vis_layout = [[sg.Text('Plot window', font=(fontname, fontsize))], \
                  [sg.Canvas(key='canvas')], \
                  [sg.Text('Up to what radius should computation be (in kpc)\n(leave blank if you chose refine box option)', font=(fontname, fontsize)), sg.InputText(20.0, key='galrad', enable_events=True, size=(10, 1), font=(fontname, fontsize)), sg.Text('OR', font=(fontname, fontsize)), sg.Checkbox('Compute only up to edge of refine box', font=(fontname, fontsize), key='fullbox', default=False)], \
                  [sg.Text('Quantity on x-axis', font=(fontname, int(fontsize/1.5))), sg.Combo(all_quantities, default_value='rad', key='xcol', enable_events=True, size=(10, 1), font=(fontname, fontsize)), sg.Text('min', font=(fontname, int(fontsize/1.5))), sg.InputText('', key='xmin', enable_events=True, size=(10, 1), font=(fontname, fontsize)), sg.Text('max', font=(fontname, int(fontsize/1.5))), sg.InputText('', key='xmax', enable_events=True, size=(10, 1), font=(fontname, fontsize))], \
                  [sg.Text('Quantity on y-axis', font=(fontname, int(fontsize/1.5))), sg.Combo(all_quantities, default_value='metal', key='ycol', enable_events=True, size=(10, 1), font=(fontname, fontsize)), sg.Text('min', font=(fontname, int(fontsize/1.5))), sg.InputText('', key='ymin', enable_events=True, size=(10, 1), font=(fontname, fontsize)), sg.Text('max', font=(fontname, int(fontsize/1.5))), sg.InputText('', key='ymax', enable_events=True, size=(10, 1), font=(fontname, fontsize))], \
                  [sg.Text('Quantity for color-coding', font=(fontname, int(fontsize/1.5))), sg.Combo(all_quantities, default_value='rad', key='colorcol', enable_events=True, size=(10, 1), font=(fontname, fontsize)), sg.Text('min', font=(fontname, int(fontsize/1.5))), sg.InputText('', key='cmin', enable_events=True, size=(10, 1), font=(fontname, fontsize)), sg.Text('max', font=(fontname, int(fontsize/1.5))), sg.InputText('', key='cmax', enable_events=True, size=(10, 1), font=(fontname, fontsize)), sg.Text('colormap (can be left blank)', font=(fontname, int(fontsize/1.5))), sg.InputText('', key='cmap', enable_events=True, size=(10, 1), font=(fontname, fontsize))], \
                  [sg.Button('Submit'), sg.Button('Cancel')]]

    vis_window = sg.Window('FOGGIE Visualisation Utility', vis_layout, finalize=True, element_justification='center', font=(fontname, fontsize), location=(180, 120), default_element_size=(20, 1), grab_anywhere=True, resizable=True)

    fig = plt.figure() # plot and get the figure instance
    draw_figure(vis_window['canvas'].TKCanvas, fig) # Add the plot to the window

    # ---------------Run the Event Loop------------------
    print('Deb267: visualisation window ready to accept changes') #
    while True:
        print('Deb272: entered event loop of visualisation window')  #
        event, values_input_dict = vis_window.read()
        print('Deb274: event and values dict read in from visualisation window, values_input_dict=', values_input_dict)  #
        for key in values_input_dict.keys():
            setattr(args, key, values_input_dict[key])
        print('Deb278: args added, args=', args)  #

        if event == 'Cancel' or event == sg.WIN_CLOSED:
            vis_window.close()
            sg.popup_error('Operation cancelled')
            break

        if event == 'Submit': # collate all args currently on screen and use it to make a datashader plot and display the plot
            print('Deb286: entered submit event of visualisation window')  #
            _, values_input_dict = vis_window.read()
            print('Deb288: _ and values dict read in after submit press, values_input_dict=', values_input_dict)  #
            for key in values_input_dict.keys():
                setattr(args, key, values_input_dict[key])
            print('Deb290: args added after submit press, args=', args)  #
            box, box_width, args = update_galrad(ds, vis_window, args)
            args = update_xparams(values_input_dict, vis_window, args)
            args = update_yparams(values_input_dict, vis_window, args)
            args = update_cparams(values_input_dict, vis_window, args)
            fig = refresh_plot(box, vis_window, args)

        elif event == 'galrad':
            box, box_width, args = update_galrad(ds, vis_window, args)
        elif event == 'xcol' or event =='xmin' or event == 'xmax':
            args = update_xparams(values_input_dict, vis_window, args)
        elif event == 'ycol' or event == 'ymin' or event == 'ymax':
            args = update_yparams(values_input_dict, vis_window, args)
        elif event == 'colorcol' or event == 'cmin' or event =='cmax' or event == 'cmap':
            args = update_cparams(values_input_dict, vis_window, args)

    # ---------Program Complete Message------------------
    sg.PopupScrolled('\n\nwindow auto closes in 10 sec (or hit OK to close)\n\n',
                     'Program is Complete!  Check your destination directory for saved files:\n',
                     f'Destination folder:  {user_input_dict["output_dir"]}',
                     font=(fontname, fontsize), title='', size=(40, 10), location=(180, 120), auto_close=True, auto_close_duration=10)

    print('GUI closed after %s' % (datetime.timedelta(minutes=(time.time() - start_time))))