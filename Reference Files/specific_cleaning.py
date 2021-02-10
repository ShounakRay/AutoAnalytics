import generic_data_prep
import pandas as pd

DATA_LOC = '/Users/Ray/Documents/Python/5 - Webber/Datasets/Student Performance/student-mat.csv'
df = pd.read_csv(DATA_LOC, sep = ';')

df = generic_data_prep.prune(df, 'True')

df['school'].replace('GP', 'Gabriel Pereira', inplace = True)
df['school'].replace('MS', 'Mousinho da Silveira', inplace = True)

df['sex'].replace('F', 'Female', inplace = True)
df['sex'].replace('M', 'Male', inplace = True)

df['address'].replace('U', 'Urban', inplace = True)
df['address'].replace('R', 'Rural', inplace = True)

df['famsize'].replace('GT3', '>3', inplace = True)
df['famsize'].replace('LE3', '<=3', inplace = True)

df['pstatus'].replace('T', 'Together', inplace = True)
df['pstatus'].replace('A', 'Apart', inplace = True)

df['medu'].replace(0, 'No education', inplace = True)
df['medu'].replace(1, 'Primary (4th grade)', inplace = True)
df['medu'].replace(2, '5-9th grade', inplace = True)
df['medu'].replace(3, 'Secondary education', inplace = True)
df['medu'].replace(4, 'Higher education', inplace = True)

df['fedu'].replace(0, 'No education', inplace = True)
df['fedu'].replace(1, 'Primary (4th grade)', inplace = True)
df['fedu'].replace(2, '5-9th grade', inplace = True)
df['fedu'].replace(3, 'Secondary education', inplace = True)
df['fedu'].replace(4, 'Higher education', inplace = True)

df['traveltime'].replace(1, '<=15 min', inplace = True)
df['traveltime'].replace(2, '15 to 30 min', inplace = True)
df['traveltime'].replace(3, '30 min. to 1 hour', inplace = True)
df['traveltime'].replace(4, '>1 hour', inplace = True)

df['studytime'].replace(1, '<2 hours', inplace = True)
df['studytime'].replace(2, '2 to 5 hours', inplace = True)
df['studytime'].replace(3, '5 to 10 hours', inplace = True)
df['studytime'].replace(4, '>10 hours', inplace = True)

df.columns = ['id', 'school_name', 'sex', 'age', 'address', 'family_size', 'parent_status', 'mother_education', 'father_education', 'mother_job', 'father_job', 'reason_for_school', 'guardian', 'travel_time', 'study_time', 'class_failures', 'extra_edu_support', 'family_edu_support', 'paid_tutors', 'extra_curic', 'nursery_school', 'higher_edu_interest', 'internet_access', 'romantic_rel_interest', 'qual_family_rel', 'free_time', 'go_out', 'work_alc_consum', 'weekend_alc_consum', 'health_status', 'num_absences', 'term_1_grade', 'term_2_grade', 'final_grade']

df.to_csv('student-mat_CLEANED.csv')

df.to_html('file.html')

#
