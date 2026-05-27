import ROOT as r
import math

def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = phi1 - phi2
    # wrap dphi to [-pi, pi]
    while dphi >  math.pi: dphi -= 2 * math.pi
    while dphi < -math.pi: dphi += 2 * math.pi
    return math.sqrt(deta * deta + dphi * dphi)

# H decays into 4900101, -4900101
f_sig = r.TFile("../Merged.root")
tree = f_sig.Get("Events")

# Histograms
h_n_matched   = r.TH1F("h_n_matched",   "Number of matched FatJets per event;N matched FatJets;Events",  5, -0.5, 4.5)
h_idx_matched   = r.TH1F("h_idx_matched",   "Indices of matched FatJets;Index;Events",  5, -0.5, 4.5)
h_jet_mass    = r.TH1F("h_jet_mass",    "Matched FatJet mass;m_{jet} [GeV];Jets",       100, 0, 300)
h_jet_msoftdrop = r.TH1F("h_jet_msoftdrop", "Matched FatJet softdrop mass;m_{SD} [GeV];Jets", 100, 0, 300)
h_jet_pt      = r.TH1F("h_jet_pt",      "Matched FatJet p_{T};p_{T} [GeV];Jets",        100, 0, 2000)
h_tagger      = r.TH1F("h_tagger",      "Matched FatJet tagger scores;Tagger score;Jets", 100, 0, 1)
h_n_sv_jet_matched = r.TH1F("h_n_sv_jet_matched", "Number of SV with #DeltaR(SV, jet) < 0.8 for dark-quark-matched jets;N_{SV};Jets", 20, -0.5, 19.5)
h_n_sv_jet_unmatched = r.TH1F("h_n_sv_jet_unmatched", "Number of SV with #DeltaR(SV, jet) < 0.8 for non-dark-quark-matched jets;N_{SV};Jets", 20, -0.5, 19.5)
h_sv_chi2     = r.TH1F("h_sv_chi2",     "SV #chi^{2} for SV with #DeltaR(SV, matched jet) < 0.8;SV #chi^{2};SV", 100, 0, 20)
h_max_sv_jet_matched = r.TH1F("h_max_sv_jet_matched", "Jet(s) with largest N_{SV} are dark-quark matched;is matched (0/1);Events", 2, -0.5, 1.5)
DARK_QUARK_PID = 4900101
DR_CUT = 0.8

for evidx, ev in enumerate(tree):
    if evidx % 10000 == 0:
        print(f"Processing event {evidx}...")
    if evidx >= 100000:  # Limit to first 10k events for testing
        break

    reco_pt = ev.FatJet_pt[0] if ev.nFatJet > 0 else 0
    if reco_pt < 300:
        continue

    nGenPart = ev.nGenPart
    pdgIds   = ev.GenPart_pdgId
    mothers  = ev.GenPart_genPartIdxMother
    eta_gen  = ev.GenPart_eta
    phi_gen  = ev.GenPart_phi

    # Find the two dark quarks (+/-4900101) that are decay products of a Higgs (pdgId==25)
    dark_quarks = []
    for i in range(nGenPart):
        if abs(pdgIds[i]) == DARK_QUARK_PID:
            mom_idx = mothers[i]
            if mom_idx >= 0 and abs(pdgIds[mom_idx]) == 25:
                dark_quarks.append(i)

    if len(dark_quarks) < 2:
        h_n_matched.Fill(0)
        continue

    # Use the first two dark quarks from H decay
    dq1, dq2 = dark_quarks[0], dark_quarks[1]
    eta_dq1, phi_dq1 = eta_gen[dq1], phi_gen[dq1]
    eta_dq2, phi_dq2 = eta_gen[dq2], phi_gen[dq2]

    nFatJet   = ev.nFatJet
    eta_fat   = ev.FatJet_eta
    phi_fat   = ev.FatJet_phi
    pt_fat    = ev.FatJet_pt
    mass_fat  = ev.FatJet_mass
    msd_fat   = ev.FatJet_msoftdrop
    xbb = ev.FatJet_globalParT3_Xbb
    xww4q = ev.FatJet_globalParT3_XWW4q
    qcd = ev.FatJet_globalParT3_QCD
    nSV = ev.nSV
    sv_eta = ev.SV_eta
    sv_phi = ev.SV_phi
    sv_chi2 = ev.SV_chi2

    jet_sv_counts = [0] * nFatJet
    for j in range(nFatJet):
        n_sv_jet = 0
        for sv_idx in range(nSV):
            dr_sv = delta_r(eta_fat[j], phi_fat[j], sv_eta[sv_idx], sv_phi[sv_idx])
            if dr_sv < DR_CUT:
                n_sv_jet += 1
        jet_sv_counts[j] = n_sv_jet

    matched_jet_indices = set()
    n_matched = 0
    for j in range(nFatJet):
        dr1 = delta_r(eta_fat[j], phi_fat[j], eta_dq1, phi_dq1)
        dr2 = delta_r(eta_fat[j], phi_fat[j], eta_dq2, phi_dq2)
        if dr1 < DR_CUT and dr2 < DR_CUT:
            matched_jet_indices.add(j)
            h_idx_matched.Fill(j)
            n_matched += 1
            h_jet_mass.Fill(mass_fat[j])
            h_jet_msoftdrop.Fill(msd_fat[j])
            h_jet_pt.Fill(pt_fat[j])
            h_tagger.Fill((xbb[j]+xww4q[j])/(xbb[j]+xww4q[j]+qcd[j]+1e-6))

            h_n_sv_jet_matched.Fill(jet_sv_counts[j])
            for sv_idx in range(nSV):
                dr_sv = delta_r(eta_fat[j], phi_fat[j], sv_eta[sv_idx], sv_phi[sv_idx])
                if dr_sv < DR_CUT:
                    h_sv_chi2.Fill(sv_chi2[sv_idx])

    for j in range(nFatJet):
        if j not in matched_jet_indices:
            h_n_sv_jet_unmatched.Fill(jet_sv_counts[j])

    if nFatJet > 0:
        max_sv = max(jet_sv_counts)
        if max_sv > 0 and len(matched_jet_indices) > 0:
            max_sv_jets = [j for j, nsv in enumerate(jet_sv_counts) if nsv == max_sv]
            is_matched = int(any(j in matched_jet_indices for j in max_sv_jets))
            h_max_sv_jet_matched.Fill(is_matched)

    h_n_matched.Fill(n_matched)

# Save histograms
f_out = r.TFile("gen_study_output.root", "RECREATE")
h_n_matched.Write()
h_jet_mass.Write()
h_jet_msoftdrop.Write()
h_jet_pt.Write()
h_tagger.Write()
h_idx_matched.Write()
h_n_sv_jet_matched.Write()
h_n_sv_jet_unmatched.Write()
h_sv_chi2.Write()
h_max_sv_jet_matched.Write()
f_out.Close()
f_sig.Close()

print("Done. Output written to gen_study_output.root")
