from __future__ import print_function, division
import os
import json
import numpy as np
import pickle
import ROOT
import pandas as pd
import argparse

import rhalphalib as rl

from hbb.common_vars import LUMI

rl.util.install_roofit_helpers()

eps=0.001 
do_systematics = False

lumi_err = {
    "2022": 1.01,
    "2023": 1.02,
    "2024": 1.02 #Dummy entry
}

def badtemp_ma(hvalues, mask=None):
    # Need minimum size & more than 1 non-zero bins           
    tot = np.sum(hvalues[mask])
    
    count_nonzeros = np.sum(hvalues[mask] > 0)
    if (tot < eps) or (count_nonzeros < 2):
        return True
    else:
        return False

def syst_variation(numerator,denominator):
    """
    Get systematic variation relative to nominal (denominator)
    """
    var = np.divide(numerator,denominator)
    var[np.where(numerator==0)] = 1
    var[np.where(denominator==0)] = 1

    return var

def smass(sName):
    if sName in ['ggF','VBF','WH','ZH','ttH']:
        _mass = 125.
    elif sName in ['Wjets','EWKW','ttbar','singlet','VV']:
        _mass = 80.379
    elif sName in ['Zjets','Zjetsbb','EWKZ','EWKZbb']:
        _mass = 91.
    else:
        raise ValueError("What is {}".format(sName))
    return _mass

def one_bin(year, tag, sName, region, ptbin, cat, syst):
    f = ROOT.TFile.Open(f"results/{tag}/{year}/signalregion.root")

    name = cat+region
    if cat == 'ggf_':
        name += 'pt'+str(ptbin)+'_'
    elif cat == 'vbf_':
        name += 'mjj'+str(ptbin)+'_'
    elif cat == 'vh_':
        name += 'pt'+str(ptbin)+'_'
    elif cat == 'mucr_':
        name += 'pt'+str(ptbin)+'_'

    name += sName+'_'+syst

    h = f.Get(name)
    newh = h.Rebin(h.GetNbinsX())
    sumw = [newh.GetBinContent(1)]
    sumw2 = [newh.GetBinError(1)]

    return (np.array(sumw), np.array([0., 1.]), "onebin", np.array(sumw2))

def get_template(year, tag, sName, region, ptbin, cat, obs, syst):
    """
    Read msd template from root file
    """
    filename = f"results/{tag}/{year}/signalregion.root"
    f = ROOT.TFile.Open(filename)

    # Only "all" category, only pt bins, only pass/fail, new naming
    name = f"all_{region}pt{ptbin}_{sName}_{syst}"

    h = f.Get(name)
    if h==None:
        raise ValueError(f"Could not find histogram {name} in file {filename}")

    sumw = []
    sumw2 = []

    for i in range(1,h.GetNbinsX()+1):
        if h.GetBinContent(i) < 0:
            sumw += [0]
            sumw2 += [0]
        else:
            sumw += [h.GetBinContent(i)]
            sumw2 += [h.GetBinError(i)*h.GetBinError(i)]

    return (np.array(sumw), obs.binning, obs.name, np.array(sumw2))

def shape_to_num(var, nom, clip=1.5):
    nom_rate = np.sum(nom)
    var_rate = np.sum(var)

    if abs(var_rate/nom_rate) > clip:
        var_rate = clip*nom_rate

    if var_rate < 0:
        var_rate = 0

    return var_rate/nom_rate

def plot_mctf(tf_MCtempl, msdbins, name,year,tag):
    """
    Plot the MC pass / fail TF as function of (pt,rho) and (pt,msd)
    """
    import matplotlib.pyplot as plt

    outdir = f"results/{tag}/{year}/plots/MCTF/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # arrays for plotting pt vs msd                    
    pts = np.linspace(450,1200,15)
    ptpts, msdpts = np.meshgrid(pts[:-1] + 0.5 * np.diff(pts), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing='ij')
    ptpts_scaled = (ptpts - 450.) / (1200. - 450.)
    rhopts = 2*np.log(msdpts/ptpts)

    rhopts_scaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhopts_scaled >= 0) & (rhopts_scaled <= 1)

    ptpts = ptpts[validbins].copy()
    msdpts = msdpts[validbins].copy()
    ptpts_scaled = ptpts_scaled[validbins].copy()
    rhopts_scaled = rhopts_scaled[validbins].copy()

    tf_MCtempl_vals = tf_MCtempl(ptpts_scaled, rhopts_scaled, nominal=True)
    df = pd.DataFrame([])
    df['msd'] = msdpts.reshape(-1)
    df['pt'] = ptpts.reshape(-1)
    df['MCTF'] = tf_MCtempl_vals.reshape(-1)

    fig, ax = plt.subplots()
    h = ax.hist2d(x=df["msd"],y=df["pt"],weights=df["MCTF"], bins=(msdbins,pts))
    plt.xlabel("$m_{sd}$ [GeV]")
    plt.ylabel("$p_{T}$ [GeV]")
    cb = fig.colorbar(h[3],ax=ax)
    cb.set_label("Ratio")
    fig.savefig(outdir + "MCTF_msdpt_"+name+".png",bbox_inches="tight")
    fig.savefig(outdir +"MCTF_msdpt_"+name+".pdf",bbox_inches="tight")
    plt.clf()

    # arrays for plotting pt vs rho                                          
    rhos = np.linspace(-6,-2.1,23)
    ptpts, rhopts = np.meshgrid(pts[:-1] + 0.5*np.diff(pts), rhos[:-1] + 0.5 * np.diff(rhos), indexing='ij')
    ptpts_scaled = (ptpts - 450.) / (1200. - 450.)
    rhopts_scaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhopts_scaled >= 0) & (rhopts_scaled <= 1)

    ptpts = ptpts[validbins].copy()
    rhopts = rhopts[validbins].copy()
    ptpts_scaled = ptpts_scaled[validbins].copy()
    rhopts_scaled = rhopts_scaled[validbins].copy()

    tf_MCtempl_vals = tf_MCtempl(ptpts_scaled, rhopts_scaled, nominal=True)

    df = pd.DataFrame([])
    df['rho'] = rhopts.reshape(-1)
    df['pt'] = ptpts.reshape(-1)
    df['MCTF'] = tf_MCtempl_vals.reshape(-1)

    fig, ax = plt.subplots()
    h = ax.hist2d(x=df["rho"],y=df["pt"],weights=df["MCTF"],bins=(rhos,pts))
    plt.xlabel("rho")
    plt.ylabel("$p_{T}$ [GeV]")
    cb = fig.colorbar(h[3],ax=ax)
    cb.set_label("Ratio")
    fig.savefig(outdir+"MCTF_rhopt_"+name+".png",bbox_inches="tight")
    fig.savefig(outdir+"MCTF_rhopt_"+name+".pdf",bbox_inches="tight")

    return

def ggfvbf_rhalphabet(args):
    """ 
    Create the data cards!
    """

    year = args.year
    tag = args.tag

    print("Running for "+year)

    working_dir = f"results/{tag}/{year}/"
    datacard_dir = f"results/{tag}/{year}/datacards/"
    initvals_dir = f"results/{tag}/{year}/initial_vals/"

    if not os.path.exists(datacard_dir):
        os.makedirs(datacard_dir)

    if not os.path.exists(initvals_dir):
        os.popen(f'cp -r initial_vals/ {initvals_dir}')

    with open(os.path.join(working_dir, 'setup.json')) as f:
        setup = json.load(f)
        cats_cfg = setup["categories"]

    total_model_bins = []

    # TT params
    tqqeffSF = rl.IndependentParameter(f'tqqeffSF_{year}', 1., -50, 50)
    tqqeffBCSF = rl.IndependentParameter(f'tqqeffBCSF_{year}', 1., -50, 50)
    tqqnormSF = rl.IndependentParameter(f'tqqnormSF_{year}', 1., -50, 50)

    sys_lumi_uncor = rl.NuisanceParameter(f'CMS_lumi_13p6TeV_{year[:4]}', 'lnN')

    #Systematics 
    sys_dict = {}
    sys_dict['pileup'] = rl.NuisanceParameter(f'CMS_PU_{year}', 'lnN')

    sys_dict['JES'] = rl.NuisanceParameter(f'CMS_scale_j_{year}', 'lnN')
    sys_dict['JER'] = rl.NuisanceParameter(f'CMS_res_j_{year}', 'lnN')
    sys_dict['UES'] = rl.NuisanceParameter(f'CMS_ues_j_{year}', 'lnN')

    sys_dict['MuonPTScale'] = rl.NuisanceParameter(f'CMS_scale_m_{year}', 'lnN')
    sys_dict['MuonPTRes'] = rl.NuisanceParameter(f'CMS_res_m_{year}', 'lnN')

    sys_dict[f'btagSFb_{year}'] = rl.NuisanceParameter(f'CMS_btagSFb_{year}', 'lnN')
    sys_dict[f'btagSFc_{year}'] = rl.NuisanceParameter(f'CMS_btagSFc_{year}', 'lnN')
    sys_dict[f'btagSFlight_{year}'] = rl.NuisanceParameter(f'CMS_btagSFlight_{year}', 'lnN')
    sys_dict['btagSFb_correlated'] = rl.NuisanceParameter(f'CMS_btagSFb_correlated_{year}', 'lnN')
    sys_dict['btagSFc_correlated'] = rl.NuisanceParameter(f'CMS_btagSFc_correlated_{year}', 'lnN')
    sys_dict['btagSFlight_correlated'] = rl.NuisanceParameter(f'CMS_btagSFlight_correlated_{year}', 'lnN')

    exp_systs = [
        # 'pileup', 
        # 'JES', 'JER', 'JER',
        # f'btagSFb_{year}',
        # f'btagSFc_{year}',
        # f'btagSFlight_{year}'
        # 'btagSFb_correlated',
        # 'btagSFc_correlated',
        # 'btagSFlight_correlated',
        # 'MuonPTScale', 'MuonPTRes'
    ]

    pdf_Higgs_ggF = rl.NuisanceParameter('pdf_Higgs_ggF','lnN')
    pdf_Higgs_VBF = rl.NuisanceParameter('pdf_Higgs_VBF','lnN')
    pdf_Higgs_VH  = rl.NuisanceParameter('pdf_Higgs_VH','lnN')
    pdf_Higgs_ttH = rl.NuisanceParameter('pdf_Higgs_ttH','lnN')

    scale_ggF = rl.NuisanceParameter('QCDscale_ggF', 'lnN')
    scale_VBF = rl.NuisanceParameter('QCDscale_VBF', 'lnN')
    scale_VH = rl.NuisanceParameter('QCDscale_VH', 'lnN')
    scale_ttH = rl.NuisanceParameter('QCDscale_ttH', 'lnN')

    isr_ggF = rl.NuisanceParameter('ISRPartonShower_ggF', 'lnN')
    isr_VBF = rl.NuisanceParameter('ISRPartonShower_VBF', 'lnN')
    isr_VH = rl.NuisanceParameter('ISRPartonShower_VH', 'lnN')
    isr_ttH = rl.NuisanceParameter('ISRPartonShower_ttH', 'lnN')

    fsr_ggF = rl.NuisanceParameter('FSRPartonShower_ggF', 'lnN')
    fsr_VBF = rl.NuisanceParameter('FSRPartonShower_VBF', 'lnN')
    fsr_VH = rl.NuisanceParameter('FSRPartonShower_VH', 'lnN')
    fsr_ttH = rl.NuisanceParameter('FSRPartonShower_ttH', 'lnN')

    validbins = {}

    msd_cfg = setup["observable"]
    msdbins = np.linspace(msd_cfg["min"], msd_cfg["max"], msd_cfg["nbins"]+1)
    msd = rl.Observable(msd_cfg["name"], msdbins)

    cats = [
        'all'
    ]

    # Only use the allowed processes
    allowed_samples = [
        'Zjetsbb', 'Zjetsc', 'Zjetslight',
        'Wjetsbb', 'Wjetsc', 'Wjetslight',
        'ttbar', 'VV'
    ]
    signal_samples = ['Zjetsbb']

    # Build qcd MC pass+fail model and fit to polynomial
    tf_params = {}
    for cat in cats:

        ptbins = np.array(cats_cfg[cat]["bins"])
        npt = len(ptbins) - 1

        # here we derive these all at once with 2D array                            
        ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing='ij')
        rhopts = 2*np.log(msdpts/ptpts)
        #ptscaled = (ptpts - 450.) / (1200. - 450.)#This is in hbb, hardcoded edges!
        ptscaled = (ptpts - ptbins[0]) / (ptbins[-1] - ptbins[0])
        rhoscaled = (rhopts - (-6.)) / ((-2.1) - (-6.)) #rho=-2.1/-6. corresponds to m/pT 0.35/0.05
        validbins[cat] = (rhoscaled >= 0.) & (rhoscaled <= 1.)
        rhoscaled[~validbins[cat]] = 1    

        tf_params[cat] = {}
        fitfailed_qcd = 0

        while fitfailed_qcd < 5:
        
            qcdmodel = rl.Model(f'qcdmodel_{cat}')
            qcdpass, qcdfail = 0., 0.

            for ptbin in range(npt):
                failCh = rl.Channel('ptbin%d%s%s%s' % (ptbin, cat, 'fail',year))
                passCh = rl.Channel('ptbin%d%s%s%s' % (ptbin, cat, 'pass',year))
                qcdmodel.addChannel(failCh)
                qcdmodel.addChannel(passCh)

                binindex = ptbin

                # QCD templates from file                           
                failTempl = get_template(year, tag, 'QCD', 'fail_', binindex+1, cat[:3]+'_', obs=msd, syst='nominal')
                passTempl = get_template(year, tag, 'QCD', f'pass_', binindex+1, cat[:3]+'_', obs=msd, syst='nominal')
                
                failCh.setObservation(failTempl, read_sumw2=True)
                passCh.setObservation(passTempl, read_sumw2=True)

                qcdfail += failCh.getObservation()[0].sum()
                qcdpass += passCh.getObservation()[0].sum()

            qcdeff = qcdpass / qcdfail
            print('Inclusive P/F from Monte Carlo = ' + str(qcdeff))


            # initial values                                         
            initF = f"results/{tag}/{year}/initial_vals/initial_vals_{cat}.json"                       
            print('Initial fit values read from file initial_vals*')
            with open(initF) as f:
                initial_vals = np.array(json.load(f)['initial_vals'])

            print("TFpf order " + str(initial_vals.shape[0]-1) + " in pT, " + str(initial_vals.shape[1]-1) + " in rho")
            tf_MCtempl = rl.BasisPoly("tf_MCtempl_"+cat+year,
                                    (initial_vals.shape[0]-1,initial_vals.shape[1]-1),
                                    ['pt', 'rho'], 
                                    basis='Bernstein',
                                    init_params=initial_vals,
                                    limits=(0, 10), 
                                    coefficient_transform=None)
            
            tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)


            for ptbin in range(npt):
                failCh = qcdmodel['ptbin%d%sfail%s' % (ptbin, cat, year)]
                passCh = qcdmodel['ptbin%d%spass%s' % (ptbin, cat, year)]
                failObs = failCh.getObservation()[0]
                
                qcdparams = np.array(
                        [
                            rl.IndependentParameter('qcdparam_ptbin%d%s%s_%d' % (ptbin, cat, year, i), 0) 
                            for i in range(msd.nbins)
                        ]
                    )
                sigmascale = 10.
                scaledparams = (
                        failObs 
                        * (1 + sigmascale/np.maximum(1., np.sqrt(failObs))) ** qcdparams
                    )
                
                fail_qcd = rl.ParametericSample(
                                'ptbin%d%sfail%s_qcd' % (ptbin, cat, year), 
                                rl.Sample.BACKGROUND, 
                                msd, 
                                scaledparams
                            )
                failCh.addSample(fail_qcd)
                pass_qcd = rl.TransferFactorSample(
                                'ptbin%d%spass%s_qcd' % (ptbin, cat, year), 
                                rl.Sample.BACKGROUND, 
                                tf_MCtempl_params[ptbin, :], 
                                fail_qcd
                            )
                passCh.addSample(pass_qcd)
                
                # drop bins outside rho validity  
                failCh.mask = validbins[cat][ptbin]
                passCh.mask = validbins[cat][ptbin]

            qcdfit_ws = ROOT.RooWorkspace('w')

            simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
            qcdfit = simpdf.fitTo(obs,
                                ROOT.RooFit.Extended(True),
                                ROOT.RooFit.SumW2Error(True),
                                ROOT.RooFit.Strategy(2),
                                ROOT.RooFit.Save(),
                                ROOT.RooFit.Minimizer('Minuit2', 'migrad'),
                                ROOT.RooFit.PrintLevel(0),
                            )
            qcdfit_ws.add(qcdfit)
            qcdfit_ws.writeToFile(os.path.join(str(datacard_dir), f'testModel_qcdfit_{cat}_{year}.root'))

            # Set parameters to fitted values  
            allparams = dict(zip(qcdfit.nameArray(), qcdfit.valueArray()))
            pvalues = []
            for i, p in enumerate(tf_MCtempl.parameters.reshape(-1)):
                p.value = allparams[p.name]
                pvalues += [p.value]
            
            if qcdfit.status() != 0:
                fitfailed_qcd += 1

                new_values = np.array(pvalues).reshape(tf_MCtempl.parameters.shape)
                print(f'Could not fit qcd, category: {cat}, new values: {new_values.tolist()}')
                with open(initF, "w") as outfile:
                    json.dump({"initial_vals":new_values.tolist()},outfile)

            else:
                break

        if fitfailed_qcd >=5:
            raise RuntimeError(f'Could not fit qcd after 5 tries')

        print("Fitted qcd for category " + cat)    

        param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
        decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + '_deco', qcdfit, param_names)
        tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)

        # Blinded TF Residual
        tf_dataResidual = rl.BasisPoly("tf_dataResidual_"+year+cat,
                                    (0,0), 
                                    ['pt', 'rho'], 
                                    basis='Bernstein',
                                    init_params=np.array([[1]]),
                                    limits=(0,20), 
                                    coefficient_transform=None)

        tf_params[cat] = qcdeff * tf_MCtempl(ptscaled, rhoscaled) * tf_dataResidual(ptscaled, rhoscaled)

    # build actual fit model now
    model = rl.Model('testModel_'+year)

    # exclude QCD from MC samps
    samps = ['Zjets','ttbar','VV']
    sigs = ['Wjets'] #Dummy signal

    for cat in cats:

        ptbins = np.array(cats_cfg[cat]["bins"])
        npt = len(ptbins) - 1

        for ptbin in range(npt):
            for region in ['pass_', 'fail_']:
                ch_name = f'ptbin{ptbin}{cat}{region.replace("_", "")}{year}'
                total_model_bins.append(ch_name)
                ch = rl.Channel(ch_name)
                model.addChannel(ch)

                templates = {}

                for sName in allowed_samples:
                    templates[sName] = get_template(year, tag, sName, region, ptbin+1, cat, obs=msd, syst='nominal')
                    nominal = templates[sName][0]

                    if(badtemp_ma(nominal)):
                        print("Sample {} is too small, skipping".format(ch.name + '_' + sName))
                        continue

                    templ = templates[sName]
                    stype = rl.Sample.SIGNAL if sName in signal_samples else rl.Sample.BACKGROUND

                    sample = rl.TemplateSample(
                        ch.name + '_' + sName,
                        stype,
                        templ,
                        force_positive=True
                    )
                    sample.setParamEffect(sys_lumi_uncor, lumi_err[year[:4]] ** (LUMI[year[:4]] / LUMI["2024"]))
                    ch.addSample(sample)

                # Data observation
                data_obs = get_template(year, tag, 'Jetdata', region, ptbin+1, cat, obs=msd, syst='nominal')
                ch.setObservation(data_obs[0:3])

    #Add data-driven qcd to model
    for cat in cats:

        ptbins = np.array(cats_cfg[cat]["bins"])

        npt = len(ptbins) - 1

        for ptbin in range(npt):

            failCh = model['ptbin%d%sfail%s' % (ptbin, cat, year)]
            passCh = model['ptbin%d%spass%s' % (ptbin, cat, year)]

            qcdparams = np.array(
                    [
                        rl.IndependentParameter('qcdparam_ptbin%d%s%s_%d' % (ptbin, cat, year, i), 0) 
                        for i in range(msd.nbins)
                    ]
                )
            initial_qcd = failCh.getObservation().astype(float)  # was integer, and numpy complained about subtracting float from it
            
            if np.any(initial_qcd < 0.):
                initial_qcd[np.where(initial_qcd<0)] = 0

            for sample in failCh:
                initial_qcd -= sample.getExpectation(nominal=True)

            if np.any(initial_qcd < 0.):
                initial_qcd[np.where(initial_qcd<0)] = 0
                raise ValueError('initial_qcd negative for some bins..', initial_qcd)

            sigmascale = 10  # to scale the deviation from initial                      
            scaledparams = (
                initial_qcd 
                * (1 + sigmascale/np.maximum(1., np.sqrt(initial_qcd))) ** qcdparams
                )

            fail_qcd = rl.ParametericSample(
                                name='ptbin%d%sfail%s_qcd' % (ptbin, cat, year), 
                                sampletype=rl.Sample.BACKGROUND, 
                                observable=msd, 
                                params=scaledparams
                            )
            failCh.addSample(fail_qcd)
            pass_qcd = rl.TransferFactorSample(
                                name='ptbin%d%spass%s_qcd' % (ptbin, cat, year), 
                                sampletype=rl.Sample.BACKGROUND, 
                                transferfactor=tf_params[cat][ptbin, :], 
                                dependentsample=fail_qcd, 
                                observable=msd
                            )
            passCh.addSample(pass_qcd)

            mask = validbins[cat][ptbin]
            failCh.mask = mask
            passCh.mask = mask
                   
    with open(os.path.join(str(datacard_dir), 'testModel_'+year+'.pkl'), 'wb') as fout:
        pickle.dump(model, fout)

    modeldir = os.path.join(str(datacard_dir), 'testModel_'+year)
    model.renderCombine(modeldir)

    out_cards = ""
    for card in total_model_bins:
        out_cards += f"{card}={card}.txt " 
    
    build_sh = os.path.join(modeldir, 'build.sh')
    with open(build_sh, "w") as f:
        f.write(f"combineCards.py {out_cards} > model_combined.txt\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix", "2024", "Run3"],
    )
    parser.add_argument(
        "--tag",
        help="tag",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    ggfvbf_rhalphabet(args)
