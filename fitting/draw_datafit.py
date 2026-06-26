import os
import ROOT
import argparse
import json

from hbb.common_vars import LUMI

blind = True

def scale_by_bin_width(hist):
    nbins = hist.GetNbinsX()
    for i in range(1, nbins + 1):  # ROOT bins start at 1
        width = hist.GetBinWidth(i)
        if width > 0:
            hist.SetBinContent(i, hist.GetBinContent(i) * width)
            hist.SetBinError(i, hist.GetBinError(i) * width)
    return hist, width

def draw(args, ptbin: int, region: str, logscale: bool = True):
    tag = args.tag
    year = args.year
    fit = args.fit

    common_dir = f"results/{tag}/{year}"

    # Load pt bin edges from setup.json
    setup_path = os.path.join(os.path.dirname(__file__), "setup.json")
    with open(setup_path, "r") as fsetup:
        setup = json.load(fsetup)
    pt_bins = setup["categories"]["all"]["bins"]
    pt_ranges = [f"{pt_bins[i]}-{pt_bins[i+1]}" for i in range(len(pt_bins)-1)]

    rZbb = 1.0
    frac = 10 # hardcoded for now for presentation
    year_string = f"{(LUMI[year] / 1000.0 / frac):0.1f}/fb, {year}"

    # Only "all" category, pt bins, pass/fail regions
    cat = "all"
    thisbin = f"pt{ptbin+1}"
    name = f"{cat}_{region}_{thisbin}_Jetdata_nominal"

    dataf = ROOT.TFile(f"{common_dir}/signalregion.root", "READ")
    if not dataf or dataf.IsZombie():
        raise RuntimeError("Could not open signalregion.root")

    data_obs = dataf.Get(name)
    if not isinstance(data_obs, ROOT.TH1D):
        raise RuntimeError(f"Could not get histogram {name} from data file")
    
    blind_min = data_obs.FindBin(110)
    blind_max = data_obs.FindBin(130)

    data_obs.SetLineColor(ROOT.kBlack)
    data_obs.SetMarkerColor(ROOT.kBlack)
    data_obs.SetMarkerStyle(20)

    if blind: 
        for i in range(blind_min, blind_max):  
            data_obs.SetBinContent(i, 0)
            data_obs.SetBinError(i, 0)

    # Fit results path
    filename = f"{common_dir}/datacards/testModel_{year}/fitDiagnosticsTest.root"
    name_plot = f"ptbin{ptbin}{cat}{region}{year}"

    histdirname = None
    if fit == "prefit":
        histdirname = f"shapes_prefit/{name_plot}/"
    elif fit == "postfit":
        histdirname = f"shapes_fit_s/{name_plot}/"
    print("histdirname:", histdirname)

    f = ROOT.TFile(filename, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open {filename}")

    # Only allowed processes
    def get_hist(proc):
        h = f.Get(histdirname + proc)
        if not h:
            h = data_obs.Clone(proc + "_empty")
            h.Reset()
        h, scale = scale_by_bin_width(h)
        return h

    # Stack order: QCD, Wjetsbb, Wjetsc, Wjetslight, Zjetsbb, Zjetsc, Zjetslight, ttbar, VV
    QCD = get_hist("qcd")
    Wjetsbb = get_hist("Wjetsbb")
    Wjetsc = get_hist("Wjetsc")
    Wjetslight = get_hist("Wjetslight")
    Zjetsbb = get_hist("Zjetsbb")
    Zjetsc = get_hist("Zjetsc")
    Zjetslight = get_hist("Zjetslight")
    ttbar = get_hist("ttbar")
    VV = get_hist("VV")

    # Set colors
    QCD.SetFillColor(ROOT.kGray)
    Wjetsbb.SetFillColor(ROOT.kGreen+2)
    Wjetsc.SetFillColor(ROOT.kGreen-3)
    Wjetslight.SetFillColor(ROOT.kGreen-7)
    Zjetsbb.SetFillColor(ROOT.kAzure-1)
    Zjetsc.SetFillColor(ROOT.kAzure+2)
    Zjetslight.SetFillColor(ROOT.kAzure+8)
    ttbar.SetFillColor(ROOT.kViolet-5)
    VV.SetFillColor(ROOT.kOrange-3)

    # Total background
    TotalBkg = f.Get(histdirname + "total_background")
    if not TotalBkg:
        raise RuntimeError(f"Could not get total_background from {histdirname}")
    TotalBkg, scale = scale_by_bin_width(TotalBkg)
    TotalBkg.SetMarkerColor(ROOT.kRed)
    TotalBkg.SetLineColor(ROOT.kRed)
    TotalBkg.SetFillColor(ROOT.kRed)
    TotalBkg.SetFillStyle(3003)

    # Mask bins and collect nonzero bin indices
    nbins = TotalBkg.GetNbinsX()
    bkg_hists = [QCD, Wjetsbb, Wjetsc, Wjetslight, Zjetsbb, Zjetsc, Zjetslight, ttbar, VV]
    nonzero_bins = []
    for i in range(1, nbins + 1):
        if TotalBkg.GetBinContent(i) == 0:
            bin_low = TotalBkg.GetBinLowEdge(i)
            bin_up = TotalBkg.GetBinLowEdge(i+1)
            pt_info = pt_ranges[ptbin] if ptbin < len(pt_ranges) else "unknown"
            print(f"Warning: TotalBkg massbin={i} [{bin_low}-{bin_up} GeV] is zero for ptbin={ptbin} [{pt_info} GeV], region={region}. Masking it.")
            data_obs.SetBinContent(i, 0)
            data_obs.SetBinError(i, 0)
            for h in bkg_hists:
                h.SetBinContent(i, 0)
                h.SetBinError(i, 0)
            TotalBkg.SetBinContent(i, 0)
            TotalBkg.SetBinError(i, 0)
        else:
            nonzero_bins.append(i)

    # Adjust x-axis range to only show nonzero bins
    if nonzero_bins:
        min_bin = nonzero_bins[0]
        max_bin = nonzero_bins[-1]
        min_x = TotalBkg.GetBinLowEdge(min_bin)
        max_x = TotalBkg.GetBinLowEdge(max_bin + 1)
        TotalBkg.GetXaxis().SetRangeUser(min_x, max_x)
    else:
        # fallback: show full range
        TotalBkg.GetXaxis().SetRangeUser(TotalBkg.GetBinLowEdge(1), TotalBkg.GetBinLowEdge(nbins + 1))

    max_val = TotalBkg.GetMaximum()
    if data_obs.GetMaximum() > max_val:
        max_val = data_obs.GetMaximum()

    TotalBkg.GetYaxis().SetRangeUser(0.001, 1000 * max_val)
    if not logscale:
        TotalBkg.GetYaxis().SetRangeUser(0, 1.3 * max_val)

    TotalBkg.GetYaxis().SetTitle(f"Events / {int(scale)} GeV")
    TotalBkg.GetXaxis().SetTitle("m_{sd} [GeV]")

    bkg = ROOT.THStack("bkg", "")
    bkg.Add(QCD)
    bkg.Add(Wjetsbb)
    bkg.Add(Wjetsc)
    bkg.Add(Wjetslight)
    bkg.Add(Zjetsbb)
    bkg.Add(Zjetsc)
    bkg.Add(Zjetslight)
    bkg.Add(ttbar)
    bkg.Add(VV)

    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetOptStat(0)

    c = ROOT.TCanvas(name_plot, name_plot, 600, 600)
    pad1 = ROOT.TPad("pad1", "pad1", 0.0, 0.33, 1.0, 1.0)
    pad2 = ROOT.TPad("pad2", "pad2", 0.0, 0.0, 1.0, 0.33)

    pad1.SetBottomMargin(1e-5)
    pad1.SetTopMargin(0.1)
    pad1.SetBorderMode(0)
    pad2.SetTopMargin(1e-5)
    pad2.SetBottomMargin(0.3)
    pad2.SetBorderMode(0)

    pad1.SetLeftMargin(0.15)
    pad2.SetLeftMargin(0.15)
    pad1.Draw()
    pad2.Draw()

    textsize1 = 16.0 / (pad1.GetWh() * pad1.GetAbsHNDC())
    textsize2 = 16.0 / (pad2.GetWh() * pad2.GetAbsHNDC())

    TotalBkg.GetYaxis().SetTitleSize(textsize1)
    TotalBkg.GetYaxis().SetLabelSize(textsize1)
    TotalBkg.GetYaxis().SetTitleOffset(2 * pad1.GetAbsHNDC())

    pad1.cd()
    if logscale:
        pad1.SetLogy()

    print("QCD:", QCD.Integral())
    print("Wjetsbb:", Wjetsbb.Integral())
    print("Wjetsc:", Wjetsc.Integral())
    print("Wjetslight:", Wjetslight.Integral())
    print("Zjetsbb:", Zjetsbb.Integral())
    print("Zjetsc:", Zjetsc.Integral())
    print("Zjetslight:", Zjetslight.Integral())
    print("ttbar:", ttbar.Integral())
    print("VV:", VV.Integral())

    TotalBkg.Draw("e2")
    bkg.Draw("histsame")
    data_obs.Draw("pesame")
    data_obs.Draw("axissame")

    # Legend
    x1, y1 = 0.6, 0.86
    leg = ROOT.TLegend(x1, y1, x1 + 0.3, y1 - 0.32)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(2)
    leg.SetTextSize(textsize1)

    leg.AddEntry(data_obs, "Data", "p")
    leg.AddEntry(TotalBkg, "Bkg. Unc.", "f")
    leg.AddEntry(QCD, "QCD", "f")
    leg.AddEntry(Wjetsbb, "W+bb", "f")
    leg.AddEntry(Wjetsc, "W+c", "f")
    leg.AddEntry(Wjetslight, "W+light", "f")
    leg.AddEntry(Zjetsbb, "Z+bb", "f")
    leg.AddEntry(Zjetsc, "Z+c", "f")
    leg.AddEntry(Zjetslight, "Z+light", "f")
    leg.AddEntry(ttbar, "ttbar", "f")
    leg.AddEntry(VV, "VV", "f")

    leg.Draw()

    l1 = ROOT.TLatex()
    l1.SetNDC()
    l1.SetTextFont(42)
    l1.SetTextSize(textsize1)
    l1.DrawLatex(0.2, 0.82, "#bf{CMS} WiP")

    l2 = ROOT.TLatex()
    l2.SetNDC()
    l2.SetTextFont(42)
    l2.SetTextSize(textsize1)
    l2.DrawLatex(0.7, 0.92, year_string)

    text3 = f"{region.capitalize()} Region"
    l3 = ROOT.TLatex()
    l3.SetNDC()
    l3.SetTextFont(42)
    l3.SetTextSize(textsize1)
    l3.DrawLatex(0.2, 0.77, text3)

    l4 = ROOT.TLatex()
    l4.SetNDC()
    l4.SetTextFont(42)
    l4.SetTextSize(textsize1)
    l4.DrawLatex(0.2, 0.72, f"all category p_{{T}} bin {ptbin+1}")

    # ratio panel
    pad2.cd()

    TotalBkg_sub = TotalBkg.Clone("TotalBkg_sub")
    TotalBkg_sub.Reset()
    data_obs_sub = data_obs.Clone("data_obs_ratio")
    data_obs_sub.Reset()

    nbins = TotalBkg_sub.GetNbinsX()
    for i in range(1, nbins + 1):
        err_data = data_obs.GetBinError(i)
        if err_data != 0:
            TotalBkg_sub.SetBinError(i, TotalBkg.GetBinError(i) / err_data)
            diff = (data_obs.GetBinContent(i) - TotalBkg.GetBinContent(i)) / err_data
            data_obs_sub.SetBinContent(i, diff)
            data_obs_sub.SetBinError(i, 1.0)
        else:
            TotalBkg_sub.SetBinError(i, 0)
            data_obs_sub.SetBinContent(i, 0)
            data_obs_sub.SetBinError(i, 0)

    TotalBkg_sub.GetYaxis().SetTitleSize(textsize2)
    TotalBkg_sub.GetYaxis().SetLabelSize(textsize2)
    TotalBkg_sub.GetXaxis().SetTitleSize(textsize2)
    TotalBkg_sub.GetXaxis().SetLabelSize(textsize2)
    TotalBkg_sub.GetYaxis().SetTitleOffset(2 * pad2.GetAbsHNDC())
    TotalBkg_sub.GetYaxis().SetTitle("(Data - Bkg)/#sigma_{Data}")
    TotalBkg_sub.SetMarkerSize(0)

    if blind:
        for i in range(blind_min, blind_max):
            data_obs.SetBinContent(i, 0)
            data_obs.SetBinError(i, 0)
            TotalBkg_sub.SetBinError(i, 0)
            data_obs_sub.SetBinContent(i, 0)
            data_obs_sub.SetBinError(i, 0)

    min2 = data_obs_sub.GetMinimum()
    max2 = data_obs_sub.GetMaximum()
    TotalBkg_sub.GetYaxis().SetRangeUser(1.3 * min2, 1.3 * max2)

    TotalBkg_sub.Draw("e2")
    data_obs_sub.Draw("pesame")

    # Save
    outdir = f"{common_dir}/plots/{fit}"
    os.makedirs(outdir, exist_ok=True)
    outpng = os.path.join(outdir, f"{name_plot}.png")
    outpdf = os.path.join(outdir, f"{name_plot}.pdf")
    c.SaveAs(outpng)
    c.SaveAs(outpdf)

    f.Close()
    dataf.Close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix", "2024"],
    )
    parser.add_argument(
        "--tag",
        help="tag",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--fit",
        help="fit",
        type=str,
        required=True,
        choices=["prefit","postfit"],
    )
    args = parser.parse_args()

    # Only "fail" and "pass" regions, three pt bins, "all" category
    for region in ["fail", "pass"]:
        for ptbin in range(3):
            draw(args, ptbin=ptbin, region=region, logscale=False)
