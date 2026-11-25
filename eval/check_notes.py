# eval/check_notes.py
import sqlite3
c = sqlite3.connect("storage_v2_copy/legal.db")
rows = c.execute("SELECT dissenters, held_line FROM case_notes "
                 "WHERE source_file LIKE 'A_K_Gopalan%'").fetchall()
print(rows)