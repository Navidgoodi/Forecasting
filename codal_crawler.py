import pandas as pd
import requests
import json
import numpy as np
from html_table_parser.parser import HTMLTableParser
import urllib.request

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def moh_tolid_foroush(symbol_list):
    """
    دریافت گزارشات ماهانه تولید و فروش از کدال
    """
    all_results = {}

    session = requests.Session()
    session.headers.update(HEADERS)

    for symb in symbol_list:
        print(f"\n{'='*50}")
        print(f"🔍 نماد: {symb}")
        print(f"{'='*50}")

        # مرحله ۱: دریافت صفحه اول برای فهمیدن تعداد صفحات
        search_url = (
            'https://search.codal.ir/api/search/v2/q?'
            '&Audited=true&AuditorRef=-1&Category=-1&Childs=false'
            '&CompanyState=0&CompanyType=1&Consolidatable=true'
            '&IsNotAudited=false&Length=-1&LetterType=58&Mains=true'
            '&NotAudited=true&NotConsolidatable=true'
            '&PageNumber=1&Publisher=false'
            f'&Symbol={symb}&TracingNo=-1&search=true'
        )

        try:
            resp_text = session.get(search_url, timeout=15).text
        except Exception as e:
            print(f"❌ خطا در اتصال: {e}")
            continue

        # استخراج تعداد صفحات و گزارش‌ها (همون روش اصلی استاد)
        parts = resp_text.split('SuperVision":{')
        
        # تعداد صفحات
        pagenum_idx = parts[0].find('"Page":')
        if pagenum_idx == -1:
            print("❌ نتیجه‌ای پیدا نشد")
            continue
        pagenum = int(parts[0][pagenum_idx + 7])

        # تعداد گزارش‌ها
        total_idx = parts[0].find('"Total":')
        page_idx = parts[0].find(',"Page":')
        report_num = int(parts[0][total_idx + 8 : page_idx])

        print(f"📊 تعداد گزارش: {report_num} | صفحات: {pagenum}")

        SaleData = np.zeros((report_num, 6))
        DateSeries = []
        cnt = 0

        # مرحله ۲: پیمایش صفحات
        for page in range(1, pagenum + 1):
            page_url = (
                'https://search.codal.ir/api/search/v2/q?'
                '&Audited=true&AuditorRef=-1&Category=-1&Childs=false'
                '&CompanyState=0&CompanyType=1&Consolidatable=true'
                '&IsNotAudited=false&Length=-1&LetterType=58&Mains=true'
                '&NotAudited=true&NotConsolidatable=true'
                f'&PageNumber={page}&Publisher=false'
                f'&Symbol={symb}&TracingNo=-1&search=true'
            )

            try:
                reqall = session.get(page_url, timeout=15).text.split('SuperVision":{')
            except Exception as e:
                print(f"⚠️ خطا صفحه {page}: {e}")
                continue

            # مرحله ۳: پیمایش گزارش‌ها
            for i in range(1, len(reqall)):
                # استخراج LetterSerial
                idx1 = reqall[i].find('aspx?LetterSerial=')
                idx2 = reqall[i].find('","HasExcel"')
                if idx1 == -1 or idx2 == -1:
                    continue

                letter_serial = reqall[i][idx1 + 18 : idx2]
                report_url = f'https://www.codal.ir/Reports/Decision.aspx?LetterSerial={letter_serial}'

                try:
                    reqnew = session.get(report_url, timeout=15).text
                except Exception as e:
                    print(f"⚠️ خطا گزارش: {e}")
                    continue

                # ---- فرمت جدید (JSON) ----
                if reqnew.find('datasource = ') > 0:
                    try:
                        raw = reqnew.split('datasource = ')[1].splitlines()[0]
                        if raw.endswith(';'):
                            raw = raw[:-1]

                        json_data = json.loads(raw)
                        sheet = json_data['sheets'][0]
                        title_fa = sheet.get('title_Fa', '')
                        cells = sheet['tables'][0]['cells'] + sheet['tables'][1]['cells']

                        # ساخت lookup dict برای سرعت
                        cell_map = {}
                        for c in cells:
                            cell_map[c['address']] = c['value']

                        # پیدا کردن ردیف‌ها و ستون تاریخ
                        row_domestic = None
                        row_export = None
                        row_total = None
                        date_col_ord = None
                        rows_a_col = []

                        for addr, val in cell_map.items():
                            if val == 'جمع فروش داخلی':
                                row_domestic = addr[1:]
                            elif val == 'جمع فروش صادراتی':
                                row_export = addr[1:]
                            elif val == 'جمع':
                                row_total = addr[1:]
                            elif val.startswith('دوره یک ماهه منتهی به ') or val.startswith('دوره 1 ماهه منتهی به '):
                                date_col_ord = ord(addr[0])
                            if addr[0] == 'A' and addr != 'A1':
                                rows_a_col.append(addr[1:])

                        if date_col_ord is None:
                            continue

                        dc = date_col_ord  # مخفف

                        def safe_get(col_offset, row):
                            key = chr(dc + col_offset) + str(row)
                            v = cell_map.get(key, '0')
                            if not v or v == '':
                                return 0
                            try:
                                return float(v)
                            except:
                                return 0

                        temp = np.zeros(6)
                        if row_domestic:
                            temp[0] = safe_get(1, row_domestic)
                            temp[1] = safe_get(3, row_domestic)
                        if row_export:
                            temp[2] = safe_get(1, row_export)
                            temp[3] = safe_get(3, row_export)
                        if row_total:
                            temp[4] = safe_get(1, row_total)
                            total_val = safe_get(3, row_total)
                            if total_val > 0:
                                temp[5] = total_val
                            else:
                                # جمع دستی
                                s = 0
                                for r in rows_a_col:
                                    key = chr(dc + 3) + r
                                    v = cell_map.get(key, '')
                                    if v and v != 'مبلغ فروش (میلیون ریال)':
                                        try:
                                            s += int(v)
                                        except:
                                            pass
                                temp[5] = s

                        # تاریخ
                        date_key = chr(dc) + '1'
                        date_raw = cell_map.get(date_key, '')
                        date_str = date_raw[22:] if len(date_raw) > 22 else ''

                        # ذخیره (بدون تکرار)
                        if date_str and date_str not in DateSeries:
                            SaleData[cnt, :] = temp
                            DateSeries.append(date_str)
                            print(f"  ✅ {title_fa} {date_str}")
                            cnt += 1

                    except Exception as e:
                        print(f"  ⚠️ خطا پارس JSON: {e}")
                        continue

                # ---- فرمت قدیمی (HTML) ----
                else:
                    try:
                        req = urllib.request.Request(url=report_url, headers=HEADERS)
                        f = urllib.request.urlopen(req)
                        dataall = f.read().decode('utf-8')

                        p = HTMLTableParser()
                        p.feed(dataall)

                        if len(p.tables) < 2:
                            continue

                        df = pd.DataFrame(p.tables[1], columns=p.tables[0][2])
                        date_str = p.tables[0][1][0][22:32]

                        if date_str not in DateSeries:
                            SaleData[cnt, 5] = float(df.iloc[-1, 5].replace(',', ''))
                            DateSeries.append(date_str)
                            print(f"  ✅ گزارش قدیمی {date_str}")
                            cnt += 1

                    except Exception as e:
                        print(f"  ⚠️ خطا پارس HTML: {e}")
                        continue

        # برش آرایه به اندازه واقعی
        SaleData = SaleData[:cnt, :]

        all_results[symb] = {
            'data': SaleData,
            'dates': DateSeries
        }

        print(f"\n📈 {symb}: {cnt} گزارش یکتا دریافت شد")

    return all_results


# ============================
#        اجرای برنامه
# ============================
if __name__ == '__main__':
    symbols = ['فولاد']  # نمادهای مورد نظر
    results = moh_tolid_foroush(symbols)

    # نمایش نتایج
    for sym, res in results.items():
        print(f"\n{'='*50}")
        print(f"نماد: {sym}")
        print(f"تعداد گزارش: {len(res['dates'])}")
        print(f"شکل داده: {res['data'].shape}")
        if len(res['dates']) > 0:
            print(f"آخرین تاریخ: {res['dates'][0]}")
            print(f"اولین تاریخ: {res['dates'][-1]}")
