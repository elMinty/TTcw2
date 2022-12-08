from math import log2

import pandas as pd


def dcg_formula(list_in):
    score = list_in[0]

    for i in range(len(list_in) - 1):
        score += list_in[i + 1] / log2(i + 2)

    return score


class Eval:

    def __init__(self):
        self.cvs_files = {}
        self.rel_df = None
        self.prec_10 = None
        self.rec_50 = None
        self.r_precision = None
        self.rel_number = None
        self.average_precision = None
        self.nDCG_10 = None
        self.nDCG_20 = None

        self.part1()

    def part1(self):
        self.__load_cvs("qrels.csv")
        self.__load_cvs("system_results.csv")

        FN = self.__combine()

        self.prec_10 = self.__precision()
        self.rec_50 = self.__recall()
        self.__r_precision()
        self.__average_precision()

        self.nDCG_10 = self.__get_nDCG(10)
        self.nDCG_20 = self.__get_nDCG(20)
        self.__to_csv()

    def __to_csv(self):

        df = pd.DataFrame(columns=['system_number', 'query_number', 'P@10', 'R@50', 'r-precision', 'AP',
                                   'nDCG@10', 'nDCG@20'])

        for i in range(6):

            for j in range(11):

                if j == 10:
                    sys_number_mean = str(i + 1)
                    q_number_mean = 'mean'
                    p_10_mean = round((sum(self.prec_10[i]) / 10) , 3)
                    r_50_mean = round((sum(self.rec_50[i])/10) , 3)
                    r_precision_mean = round((sum(self.r_precision[i]) / 10) , 3)
                    ap_mean = round((sum(self.average_precision[i]) / 10) ,3)
                    n_10_mean = round((sum(self.nDCG_10[i]) / 10), 3)
                    n_20_mean = round((sum(self.nDCG_20[i]) / 10), 3)

                    d = {'system_number': sys_number_mean, 'query_number': q_number_mean, 'P@10': p_10_mean, 'R@50': r_50_mean,
                         'r-precision': r_precision_mean, 'AP': ap_mean,
                         'nDCG@10': n_10_mean, 'nDCG@20': n_20_mean}

                    df = df.append(d, ignore_index=True)

                    continue

                # sys_number

                sys_number = str(i + 1)

                # q number

                q_number = str(j + 1)

                # P@10

                p_10 = round(self.prec_10[i][j], 3)

                # R@50

                r_50 = round(self.rec_50[i][j],3)

                # r-precision

                r_p = round(self.r_precision[i][j],3)

                # AP

                a_p = round(self.average_precision[i][j],3)

                # nDCG@10

                n_10 = round(self.nDCG_10[i][j],3)

                # nDCG@20

                n_20 = round(self.nDCG_20[i][j],3)

                d = {'system_number': sys_number, 'query_number': q_number, 'P@10': p_10, 'R@50': r_50,
                     'r-precision': r_p, 'AP': a_p,
                     'nDCG@10': n_10, 'nDCG@20': n_20}

                df = df.append(d, ignore_index=True)


        df.to_csv('ir_eval.csv', index=False)

    def __get_nDCG(self, cut_off):

        nDCG = []

        df = self.rel_df.copy()

        for i in range(6):
            sys = []
            df_sys = df.loc[df['system_number'] == i + 1]

            for j in range(10):

                df_q = df_sys.loc[(df_sys['query_number'] == j + 1)]
                dcg = df_q.loc[(df_q['rank_of_doc'] <= cut_off)]
                dcg = dcg['Relevancy score'].tolist()
                iDCG = self.cvs_files['qrels.csv']
                iDCG = iDCG.loc[iDCG['query_id'] == j+1]
                iDCG = iDCG['relevance'].tolist()
                iDCG.sort(reverse=True)

                dcg_score = dcg_formula(dcg)
                iDCG_score = dcg_formula(iDCG[0:cut_off])

                if iDCG_score == 0:
                    sys.append(0.0)
                else:
                    sys.append(dcg_score / iDCG_score)

            nDCG.append(sys)

        return nDCG

    def __average_precision(self):

        avg_prec = []

        df = self.rel_df.copy()

        for i in range(6):

            sys = []

            df_sys = df.loc[df['system_number'] == i + 1]

            # for each system go through different queries

            for j in range(10):

                rel_rank = []

                df_q = df_sys.loc[(df_sys['query_number'] == j + 1) & (df_sys['Relevant'] == True)]

                ranks = df_q['rank_of_doc'].tolist()

                if ranks:

                    for k in range(len(ranks)):
                        rel_rank.append((k + 1) / ranks[k])

                    sys.append(sum(rel_rank) / self.rel_number[j])
                else:
                    sys.append(0.0)

                avg_prec.append(sys)

        self.average_precision = avg_prec

    def __r_precision(self):

        r_prec = []

        df = self.rel_df.copy()

        for i in range(6):

            sys = []

            df_sys = df.loc[df['system_number'] == i + 1]

            # for each system go through different queries

            for j in range(10):
                df_q = df_sys.loc[df_sys['query_number'] == j + 1]

                cutoff = self.rel_number[j]

                relevant = len(df_q.loc[(df_q['Relevant'] == True) & (df_q['rank_of_doc'] <= cutoff)])

                sys.append(relevant / cutoff)

            r_prec.append(sys)

        self.r_precision = r_prec

    def __recall(self):

        recall_at_50 = []

        df = self.rel_df.copy()

        # get top 50 ranked documents

        df_50 = df.loc[df['rank_of_doc'] <= 50]

        print(df_50['rank_of_doc'])

        for i in range(6):

            # 50 top rank
            df_sys_50 = df_50.loc[df['system_number'] == i + 1]
            # whole returned sys
            df_sys = df.loc[df['system_number'] == i + 1]

            rec_sys = []

            # for each system go through different queries

            for j in range(10):

                rec_q = []

                # how many out of the top ten ranked are relevant

                rel_doc = df_sys.loc[(df_sys['query_number'] == j + 1) & (df_sys['Relevant'] == True)]

                rel_doc = len(rel_doc)

                rel_doc_50 = len(df_sys_50.loc[(df_sys_50['query_number'] == j + 1) & (df_sys_50['Relevant'] == True)])

                if rel_doc == 0:
                    rec_sys.append(0.0)

                else:
                    rec_sys.append(rel_doc_50 / self.rel_number[j])

            recall_at_50.append(rec_sys)

        return recall_at_50

    def __precision(self):

        precision_at_ten = []

        df = self.rel_df.copy()

        # get top 10 ranked result data frame

        df = df.loc[df['rank_of_doc'] <= 10]

        # go through different systems

        for i in range(6):

            df_sys = df.loc[df['system_number'] == i + 1]

            prec_sys = []

            # for each system go through different queries

            for j in range(10):
                prec_q = []

                # how many out of the top ten ranked are relevant

                df_sys_q = df_sys.loc[(df_sys['query_number'] == j + 1) & (df_sys['Relevant'] == True)]

                prec_sys.append(len(df_sys_q) / 10)

            precision_at_ten.append(prec_sys)

        return precision_at_ten

    def __load_cvs(self, file_name):

        df = pd.read_csv(file_name)

        self.cvs_files[file_name] = df

    def __combine(self):

        relevant = self.cvs_files.get("qrels.csv")
        result_docs = self.cvs_files.get("system_results.csv")

        if relevant is None:
            print("qrels.csv not loaded")
            raise Exception
        if result_docs is None:
            (print("system_results.csv not loaded"))
            raise Exception

        # get number of relevant documents in qrels
        rel_no = []

        for i in range(10):
            rel_number = len(relevant.loc[relevant['query_id'] == i + 1])
            rel_no.append(rel_number)

        self.rel_number = rel_no

        # add relevancy column to results and then relevancy score (new df)

        df_rel = result_docs.copy()

        row_number = len(result_docs.index)

        relevant_row = [False] * row_number
        relevancy_score_row = [0] * row_number

        df_rel.insert(5, 'Relevant', relevant_row)
        df_rel.insert(6, 'Relevancy score', relevancy_score_row)

        # go through qrels, take the query,doc,score, match in df_rel, changing relevant to TRUE if present

        rel_doc_query = []

        for index, row in relevant.iterrows():
            rel_doc_query.append((row['query_id'], row['doc_id'], row['relevance']))

        FN = []

        for entry in rel_doc_query:

            query, doc, rel = entry

            if df_rel.loc[(df_rel['query_number'] == query) & (df_rel['doc_number'] == doc)].empty:
                FN.append((query, doc))

            else:
                df_rel.loc[(df_rel['query_number'] == query) & (df_rel['doc_number'] == doc), 'Relevant'] = True
                df_rel.loc[(df_rel['query_number'] == query) & (df_rel['doc_number'] == doc), 'Relevancy score'] = rel

        self.rel_df = df_rel.copy()

        return FN

        # above df_rel has all returned results with whether they are relevant
        # FN has all the docs that are relevant but never turned up in the results
        # TP has all the docs that appear in both
        # FP has the docs that appear in the results but not 


if __name__ == "__main__":
    e = Eval()
    exit()
