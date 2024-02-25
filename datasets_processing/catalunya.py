# preprocesing
import pandas as pd

from datasets_processing._dataset import BaseDataset, get_the_middle

data_mapping = {'V56_@R1_resum_risc_global_reverse': {'Moderada':2, 'Alta':3, 'Baixa':1},
'V57_@R2_resum_risc_delictes_violents': {'Moderada':2, 'Alta':3, 'notapplicable':0, 'Baixa':1},
'V58_@R3_resum_risc_violencia_centre': {'Moderada':2, 'notapplicable':0, 'Baixa':1, 'Alta':3},
'V59_@R4_resum_risc_sortides_permisos': {'Moderada':2, 'notapplicable':0, 'Alta':3, 'Baixa': 1},
'V65_@1_violencia_previa': {'Alt': 3, 'Baix': 1, 'Moderat': 2},
'V66_@2_historia_delictes_no_violents': {'Moderat': 2, 'Alt': 3, 'Baix': 1},
'V67_@3_inici_precoç_violencia': {'Alt': 3, 'Moderat': 2, 'Baix': 1},
'V68_@4_fracas_intervencions_anteriors': {'Baix': 1, 'Alt': 3, 'Moderat': 2},
'V69_@5_intents_autolesio_suicidi_anteriors': {'Moderat': 2, 'Alt': 3, 'Baix': 1},
'V70_@6_exposicio_violencia_llar': {'Moderat': 2, 'Alt': 3, 'Baix': 1},
'V71_@7_historia_maltracte_infantil': {'Baix': 1, 'Moderat': 2, 'Alt': 3},
'V72_@8_delinquencia_pares': {'Alt': 3, 'Baix': 1, 'Moderat': 2,},
'V73_@9_separacio_precoç_pares': {'Alt': 3, 'Baix': 1, 'Moderat': 2},
'V74_@10_baix_rendiment_escola': {'Alt': 3, 'Moderat': 2, 'Baix': 1},
'V75_@11_delinquencia_grup_iguals': {'Moderat': 2, 'Alt': 3, 'Baix': 1},
'V76_@12_rebuig_grup_iguals': {'Moderat': 2, 'Baix': 1, 'Alt': 3},
'V77_@13_estrés_incapacitat_enfrontar_dificultats': {'Moderat': 2, 'Alt': 3, 'Baix': 1},
'V78_@14_escassa_habilitat_pares_educar': {'Moderat': 2, 'Alt': 3, 'Baix': 1},
'V79_@15_manca_suport_personal_social': {'Moderat': 2, 'Baix': 1, 'Alt': 3},
'V80_@16_entorn_marginal': {'Moderat': 2, 'Alt': 3, 'Baix': 1},
'V81_@17_actitud_negatives': {'Moderat': 2, 'Alt': 3, 'Baix': 1},
'V82_@18_assumpcio_riscos_impulsivitat': {'Moderat': 2, 'Alt': 3, 'Baix': 1},
'V83_@19_problemes_consum_toxics': {'Alt': 3, 'Moderat': 2, 'Baix': 1},
'V84_@20_problemes_maneig_enuig': {'Alt': 3, 'Moderat': 2, 'Baix': 1},
'V85_@21_baix_nivell_empatia_remordiment': {'Moderat': 2, 'Alt': 3, 'Baix': 1},
'V86_@22_problemes_concentracio_hiperactivitat': {'Alt': 3, 'Baix': 1, 'Moderat': 2},
'V87_@23_baixa_colaboracio_intervencions': {'Moderat': 2, 'Baix': 1, 'Alt': 3},
'V88_@24_baix_compromis_escolar_laboral': {'Moderat': 2, 'Baix': 1, 'Alt': 3},
'V89_@P1_impicacio_prosocial': {'Present':1, 'Absent':0},
'V90_@P2_suport_social_fort': {'Present':1, 'Absent':0},
'V91_@P3_forta_vinculacio_adult_prosocial': {'Present':1, 'Absent':0},
'V92_@P4_actitud_positiva_intervencions_autoritat': {'Present':1, 'Absent':0},
'V93_@P5_fort_compromis_escola_treball': {'Present':1, 'Absent':0},
'V94_@P6_perseverança_tret_personalitat': {'Present':1, 'Absent':0}}


class CatalunyaDataset(BaseDataset):
    def __init__(self, att, data_dir=None, random_seed=0, outcome_type='binary'):
        self.outcome_type = outcome_type
        self._outcome_original = 'binary'
        self._name = 'catalunya'
        self._att = att
        self.preprocess()

    def preprocess(self):

        data = pd.read_csv("datasets/catalunya/df_savry.csv", index_col=0)

        all_columns = ['sex', 'foreigner', 'national_group', 'province_resid',
         'age_atmaincrime', 'dateofbirth', 'priorcrimes_dummy',
         'priorcrimes_number_group', 'maincrime_cat', 'maincrime_violent',
         'totalmaincrimes_number', 'maincrime_date', 'province_execution',
         'maincrime_program', 'maincrime_program2', 'maincrime_program_group',
         'maincrime_programduration_group', 'V28_temps_inici',
         'V29_durada_programa', 'V30_data_inici_programa',
         'V31_data_fi_programa', 'V54_SAVRYprograma', 'V55_SAVRYdata',
         'V56_@R1_resum_risc_global_reverse',
         'V57_@R2_resum_risc_delictes_violents',
         'V58_@R3_resum_risc_violencia_centre',
         'V59_@R4_resum_risc_sortides_permisos', 'V60_SAVRY_total_score',
         'V61_SAVRY_historics_total_score', 'V62_SAVRY_socials_total_score',
         'V63_SAVRY_individuals_total_score', 'V64_SAVRY_proteccio_total_score',
         'V65_@1_violencia_previa', 'V66_@2_historia_delictes_no_violents',
         'V67_@3_inici_precoç_violencia',
         'V68_@4_fracas_intervencions_anteriors',
         'V69_@5_intents_autolesio_suicidi_anteriors',
         'V70_@6_exposicio_violencia_llar', 'V71_@7_historia_maltracte_infantil',
         'V72_@8_delinquencia_pares', 'V73_@9_separacio_precoç_pares',
         'V74_@10_baix_rendiment_escola', 'V75_@11_delinquencia_grup_iguals',
         'V76_@12_rebuig_grup_iguals',
         'V77_@13_estrés_incapacitat_enfrontar_dificultats',
         'V78_@14_escassa_habilitat_pares_educar',
         'V79_@15_manca_suport_personal_social', 'V80_@16_entorn_marginal',
         'V81_@17_actitud_negatives', 'V82_@18_assumpcio_riscos_impulsivitat',
         'V83_@19_problemes_consum_toxics', 'V84_@20_problemes_maneig_enuig',
         'V85_@21_baix_nivell_empatia_remordiment',
         'V86_@22_problemes_concentracio_hiperactivitat',
         'V87_@23_baixa_colaboracio_intervencions',
         'V88_@24_baix_compromis_escolar_laboral', 'V89_@P1_impicacio_prosocial',
         'V90_@P2_suport_social_fort',
         'V91_@P3_forta_vinculacio_adult_prosocial',
         'V92_@P4_actitud_positiva_intervencions_autoritat',
         'V93_@P5_fort_compromis_escola_treball',
         'V94_@P6_perseverança_tret_personalitat', 'V95_FACT1mean_ANTISOCIAL',
         'V96_FACT2mean_DINAMICAFAM', 'V97_FACT3mean_PERSONALITAT',
         'V98_FACT4mean_SUPORTSOCIAL', 'V99_FACT5mean_SUSCEPTIBILITAT',
         'recid15_date', 'recid15_age', 'recid15_number', 'recid15_durat_group',
         'recid15_crime_group', 'recid15_crime_viol', 'recid15_program',
         'recid15_program_group', 'recid15_totsum', 'recid15_dummy',
         'recid13_date', 'recid13_age', 'recid13_number', 'recid13_durat_group',
         'recid13_crime_group', 'recid13_crime_viol', 'recid13_program',
         'recid13_program_group', 'recid13_totsum', 'recid13_dummy',
         'maincrime_year', 'recid15_viol']

        label_column = 'recid15_dummy'
        protected_features = ['sex', 'foreigner', 'national_group']

        column_mapping = {'sex': {'Home': 0, 'Dona' : 1},
                        'foreigner': {'Espanyol':1, 'Estranger': 0},
                        'national_group': {'Spain':1, 'Magrib':0, 'Centre i Sud Amèrica':0, 'Altres':0, 'Europa':1},
                        'recid15_dummy': {'No': 1, 'Sí': 0}}


        df_protected = pd.DataFrame()
        # Convert columns to numeric, if possible
        for col, mapping in column_mapping.items():
            df_protected[col] = data[col].map(mapping)

        df_cat = pd.DataFrame()
        # Convert columns to numeric, if possible
        for col, mapping in data_mapping.items():
            df_cat[col] = data[col].map(mapping)

        drop_columns =['province_resid','age_atmaincrime', 'priorcrimes_dummy', 'priorcrimes_number_group','maincrime_cat',
                       'maincrime_violent', 'totalmaincrimes_number','province_execution', 'maincrime_program2',
                       'maincrime_programduration_group', 'V28_temps_inici', 'V29_durada_programa','recid13_dummy',
                        'recid13_date','recid13_age','recid13_number','recid13_program','recid13_program_group',
                       'recid13_totsum', 'recid15_date', 'recid15_age','recid15_number', 'recid15_durat_group',
                       'recid15_crime_group','recid15_crime_viol', 'recid15_program', 'recid15_program_group',
                       'recid15_totsum', 'recid13_durat_group','maincrime_date','recid13_crime_group', 'recid13_crime_viol',
                       'dateofbirth','maincrime_program','maincrime_program_group','V30_data_inici_programa',
                       'V31_data_fi_programa','V55_SAVRYdata','recid15_viol']


        categorical_columns = [
                          'V56_@R1_resum_risc_global_reverse', 'V57_@R2_resum_risc_delictes_violents',
                          'V58_@R3_resum_risc_violencia_centre', 'V59_@R4_resum_risc_sortides_permisos',
                          'V60_SAVRY_total_score', 'V61_SAVRY_historics_total_score', 'V62_SAVRY_socials_total_score',
                          'V63_SAVRY_individuals_total_score', 'V64_SAVRY_proteccio_total_score',
                          'V65_@1_violencia_previa', 'V66_@2_historia_delictes_no_violents',
                          'V67_@3_inici_precoç_violencia', 'V68_@4_fracas_intervencions_anteriors',
                          'V69_@5_intents_autolesio_suicidi_anteriors', 'V70_@6_exposicio_violencia_llar',
                          'V71_@7_historia_maltracte_infantil', 'V72_@8_delinquencia_pares',
                          'V73_@9_separacio_precoç_pares', 'V74_@10_baix_rendiment_escola',
                          'V75_@11_delinquencia_grup_iguals', 'V76_@12_rebuig_grup_iguals',
                          'V77_@13_estrés_incapacitat_enfrontar_dificultats', 'V78_@14_escassa_habilitat_pares_educar',
                          'V79_@15_manca_suport_personal_social', 'V80_@16_entorn_marginal',
                          'V81_@17_actitud_negatives', 'V82_@18_assumpcio_riscos_impulsivitat',
                          'V83_@19_problemes_consum_toxics', 'V84_@20_problemes_maneig_enuig',
                          'V85_@21_baix_nivell_empatia_remordiment', 'V86_@22_problemes_concentracio_hiperactivitat',
                          'V87_@23_baixa_colaboracio_intervencions', 'V88_@24_baix_compromis_escolar_laboral',
                          'V89_@P1_impicacio_prosocial', 'V90_@P2_suport_social_fort',
                          'V91_@P3_forta_vinculacio_adult_prosocial',
                          'V92_@P4_actitud_positiva_intervencions_autoritat', 'V93_@P5_fort_compromis_escola_treball',
                          'V94_@P6_perseverança_tret_personalitat']



        data.drop(drop_columns, axis=1, inplace=True)

        num_cols = ['V60_SAVRY_total_score', 'V61_SAVRY_historics_total_score', 'V62_SAVRY_socials_total_score','V63_SAVRY_individuals_total_score', 'V64_SAVRY_proteccio_total_score']
        df_num = pd.DataFrame(data[num_cols], columns=num_cols)
        for c in num_cols:
            df_num[c] = df_num[c].astype(int)

        df_base = pd.concat([df_cat, df_num, df_protected], axis=1)
        df_base.rename(columns={"recid15_dummy": "recid"}, inplace=True)
        df_base.rename(columns={"national_group": "NatG"}, inplace=True)

        df_base.dropna(axis=0, how='any', inplace=True)
        df_base.reset_index(drop=True, inplace=True)


        # this is always binary 'No': 1, 'Sí': 0
        df_base['binary_recid'] = df_base['recid']

        target_variable_ordinal = 'recid'
        target_variable_binary = 'binary_recid'

        self._ds = df_base

        self._explanatory_variables = ['V56_@R1_resum_risc_global_reverse',
                                       'V57_@R2_resum_risc_delictes_violents',
                                       'V58_@R3_resum_risc_violencia_centre',
                                       'V59_@R4_resum_risc_sortides_permisos', 'V65_@1_violencia_previa',
                                       'V66_@2_historia_delictes_no_violents', 'V67_@3_inici_precoç_violencia',
                                       'V68_@4_fracas_intervencions_anteriors',
                                       'V69_@5_intents_autolesio_suicidi_anteriors',
                                       'V70_@6_exposicio_violencia_llar', 'V71_@7_historia_maltracte_infantil',
                                       'V72_@8_delinquencia_pares', 'V73_@9_separacio_precoç_pares',
                                       'V74_@10_baix_rendiment_escola', 'V75_@11_delinquencia_grup_iguals',
                                       'V76_@12_rebuig_grup_iguals',
                                       'V77_@13_estrés_incapacitat_enfrontar_dificultats',
                                       'V78_@14_escassa_habilitat_pares_educar',
                                       'V79_@15_manca_suport_personal_social', 'V80_@16_entorn_marginal',
                                       'V81_@17_actitud_negatives', 'V82_@18_assumpcio_riscos_impulsivitat',
                                       'V83_@19_problemes_consum_toxics', 'V84_@20_problemes_maneig_enuig',
                                       'V85_@21_baix_nivell_empatia_remordiment',
                                       'V86_@22_problemes_concentracio_hiperactivitat',
                                       'V87_@23_baixa_colaboracio_intervencions',
                                       'V88_@24_baix_compromis_escolar_laboral', 'V89_@P1_impicacio_prosocial',
                                       'V90_@P2_suport_social_fort',
                                       'V91_@P3_forta_vinculacio_adult_prosocial',
                                       'V92_@P4_actitud_positiva_intervencions_autoritat',
                                       'V93_@P5_fort_compromis_escola_treball',
                                       'V94_@P6_perseverança_tret_personalitat', 'V60_SAVRY_total_score',
                                       'V61_SAVRY_historics_total_score', 'V62_SAVRY_socials_total_score',
                                       'V63_SAVRY_individuals_total_score', 'V64_SAVRY_proteccio_total_score',
                                       'sex', 'foreigner', 'NatG']

        if self.outcome_type == 'binary':
            self._outcome_label = target_variable_binary
        else:
            self._outcome_label = target_variable_ordinal  # [1-9]

        self._favorable_label_binary = [1]
        self._favorable_label_continuous = [1]

        # 'sex': {'Home': 0, 'Dona': 1},
        # 'foreigner': {'Espanyol': 1, 'Estranger': 0},
        # 'national_group': {'Spain': 1, 'Magrib': 0, 'Centre i Sud Amèrica': 0, 'Altres': 0, 'Europa': 1},

        self._protected_att_name = [self._att]
        self._privileged_classes = [[1]]
        self._privileged_groups = [{self._att: 1}]
        self._unprivileged_groups = [{self._att: 0}]

        self._continuous_label_name = target_variable_ordinal
        self._binary_label_name = target_variable_binary

        self._cut_point = 1
        self._non_favorable_label_continuous = [0]

        self._fav_dict = self.assign_ranges_to_ordinal(fav=True)
        self._nonf_dict = self.assign_ranges_to_ordinal(fav=False, sort_desc=True)
        self._middle_fav = get_the_middle(self.favorable_label_continuous)
        self._middle_non_fav = get_the_middle(self.non_favorable_label_continuous)
