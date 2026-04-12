"""
BIST-specific evaluation questions.
Minimum 10 questions covering all four question types from the assignment.
"""

EVAL_QUESTIONS = [
    # ── KAP-Centric Questions (4.2.2.1) ──────────────────────────────────────
    {
        "id": "kap_001",
        "question": "ASELS son 6 ayda hangi tür KAP açıklamaları yaptı?",
        "question_en": "What types of KAP disclosures has ASELS published in the last 6 months?",
        "ticker": "ASELS",
        "category": "kap_centric",
        "expected_sources": ["kap"],
        "expected_keywords": ["özel durum", "yönetim kurulu", "finansal", "KAP"],
        "ground_truth": (
            "ASELS son 6 ayda özel durum açıklamaları, yönetim kurulu kararları ve "
            "finansal tablo açıklamaları yapmıştır."
        ),
    },
    {
        "id": "kap_002",
        "question": "GARAN'ın son KAP açıklamalarında öne çıkan konular nelerdir?",
        "question_en": "What are the key topics in GARAN's recent KAP disclosures?",
        "ticker": "GARAN",
        "category": "kap_centric",
        "expected_sources": ["kap"],
        "expected_keywords": ["finansal", "temettü", "yönetim"],
        "ground_truth": "GARAN'ın son KAP açıklamaları sermaye artırımı, temettü ve banka finansal sonuçlarını kapsamaktadır.",
    },
    {
        "id": "kap_003",
        "question": "THYAO için son 3 ayda kaç adet özel durum açıklaması yapıldı?",
        "question_en": "How many material event disclosures did THYAO publish in the last 3 months?",
        "ticker": "THYAO",
        "category": "kap_centric",
        "expected_sources": ["kap"],
        "expected_keywords": ["özel durum", "açıklama"],
        "ground_truth": "Dönemsel açıklama sayısı değişkendir.",
    },
    # ── Brokerage Narrative Questions (4.2.2.2) ───────────────────────────────
    {
        "id": "brokerage_001",
        "question": "AKBNK için son araştırma raporlarında hangi ortak temalar öne çıkıyor?",
        "question_en": "What common themes appear across recent brokerage reports for AKBNK?",
        "ticker": "AKBNK",
        "category": "brokerage_narrative",
        "expected_sources": ["brokerage"],
        "expected_keywords": ["kredi büyümesi", "net faiz marjı", "dijital bankacılık"],
        "ground_truth": "Analist raporları kredi büyümesi, faiz marjı ve dijital bankacılık dönüşümünü ön plana çıkarmaktadır.",
    },
    {
        "id": "brokerage_002",
        "question": "EREGL ile ilgili aracı kurum raporlarında öne çıkan sektörel temalar nelerdir?",
        "question_en": "What sector themes appear in brokerage reports about EREGL?",
        "ticker": "EREGL",
        "category": "brokerage_narrative",
        "expected_sources": ["brokerage"],
        "expected_keywords": ["çelik", "demir", "ihracat", "kapasite"],
        "ground_truth": "Çelik sektör analizi, kapasite kullanımı ve ihracat dinamikleri öne çıkmaktadır.",
    },
    # ── News vs Disclosure Consistency (4.2.2.3) ──────────────────────────────
    {
        "id": "consistency_001",
        "question": "BIMAS hakkındaki son haberler resmi KAP açıklamalarıyla tutarlı mı?",
        "question_en": "Do recent news articles about BIMAS contradict or align with official KAP disclosures?",
        "ticker": "BIMAS",
        "category": "consistency",
        "expected_sources": ["kap", "news"],
        "expected_keywords": ["tutarlı", "çelişki", "KAP", "haber"],
        "ground_truth": "KAP açıklamaları ve haberler karşılaştırılarak tutarlılık analizi yapılır.",
    },
    {
        "id": "consistency_002",
        "question": "TUPRS için medya haberleri ile KAP açıklamaları arasında önemli bir fark var mı?",
        "question_en": "Is there a significant discrepancy between media coverage and KAP disclosures for TUPRS?",
        "ticker": "TUPRS",
        "category": "consistency",
        "expected_sources": ["kap", "news"],
        "expected_keywords": ["medya", "KAP", "fark", "tutarlı"],
        "ground_truth": "Medya ve KAP karşılaştırması çerçevesinde tutarlılık veya çelişki belirlenir.",
    },
    # ── Narrative Evolution (4.2.2.4) ─────────────────────────────────────────
    {
        "id": "narrative_001",
        "question": "KCHOL etrafındaki anlatı son 6 ayda nasıl değişti?",
        "question_en": "How has the narrative around KCHOL changed over the last 6 months?",
        "ticker": "KCHOL",
        "category": "narrative_evolution",
        "expected_sources": ["news", "brokerage"],
        "expected_keywords": ["değişim", "anlatı", "tempo", "holding"],
        "ground_truth": "Zaman içindeki anlatı değişimleri haber ve araştırma raporları çerçevesinde analiz edilir.",
    },
    {
        "id": "narrative_002",
        "question": "PGSUS ile ilgili haberlerin tonu geçen yıla kıyasla nasıl değişti?",
        "question_en": "How has the tone of news about PGSUS changed compared to last year?",
        "ticker": "PGSUS",
        "category": "narrative_evolution",
        "expected_sources": ["news"],
        "expected_keywords": ["ton", "değişim", "havacılık", "yolcu"],
        "ground_truth": "Havacılık sektörü haberciliğinin ton değişimi analiz edilir.",
    },
    # ── General Market Intelligence ───────────────────────────────────────────
    {
        "id": "market_001",
        "question": "Türk bankacılık sektörüne ilişkin son açıklamalar ne söylüyor?",
        "question_en": "What do recent disclosures say about the Turkish banking sector?",
        "ticker": "",
        "category": "market_intelligence",
        "expected_sources": ["kap", "news", "brokerage"],
        "expected_keywords": ["bankacılık", "faiz", "kredi", "sermaye"],
        "ground_truth": "Bankacılık sektörü KAP açıklamaları ve araştırma raporları çerçevesinde analiz edilir.",
    },
    {
        "id": "market_002",
        "question": "BIST savunma sektöründe son dönemde hangi gelişmeler öne çıktı?",
        "question_en": "What recent developments have stood out in BIST's defense sector?",
        "ticker": "ASELS",
        "category": "market_intelligence",
        "expected_sources": ["kap", "news"],
        "expected_keywords": ["savunma", "sipariş", "ihracat", "ASELSAN"],
        "ground_truth": "Savunma sektörü son gelişmeleri KAP ve haber kaynakları üzerinden değerlendirilir.",
    },
]
